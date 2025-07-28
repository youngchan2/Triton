import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import tensorrt as trt
import tempfile
import os
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache

device = 'cuda'
dtype = torch.float16

class SimpleAttention(nn.Module):
    """Simple manual attention implementation for comparison"""
    def __init__(self, M, N, D, P, cache_K, cache_V):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.weight = nn.Parameter(torch.ones(self.N))
        self.W_total = nn.Linear(self.N, 3*self.N, bias=False)

        # Random Q, K, V for testing
        self.q = torch.randn((self.H, M, D), device='cuda', dtype=torch.float16)
        self.cache_K = cache_K  # (H, P+M, D)
        self.cache_V = cache_V  # (H, P+M, D)

    def forward(self, X):
        scale_factor = 1.0 / math.sqrt(self.D)
        

        X_norm = F.rms_norm(X, normalized_shape=(self.N,), weight=self.weight)
        
        qkv = self.W_total(X_norm)
        q, k, v = torch.split(qkv, self.N, dim=1)
        q = q.view(1, self.M, self.H, self.D)
        k = k.view(1, self.M, self.H, self.D)
        v = v.view(1, self.M, self.H, self.D)

        # Compute attention for each head
        output_heads = []
        for h in range(self.H):
            # Q: (M, D), K: (P+M, D), V: (P+M, D)
            scores = torch.matmul(self.q[h], self.cache_K[h].transpose(0, 1)) * scale_factor  # (M, P+M)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, self.cache_V[h])  # (M, D)
            output_heads.append(output)
        
        # Concatenate heads
        output = torch.cat(output_heads, dim=-1)  # (M, H*D) = (M, N)
        return output

class Torch_kernel1(nn.Module):
    def __init__(self, H, D, num_heads, cache_K, cache_V, context_length):
        super(Torch_kernel1, self).__init__()
        self.H = H
        self.D = D
        self.num_heads = num_heads

        self.weight = nn.Parameter(torch.ones(self.H))
        self.W_total = nn.Linear(self.H, 3*self.H, bias=False)

        self.cache_K = cache_K 
        self.cache_V = cache_V

        self.context_length = context_length

    def forward(self, X):
        seq_len, H = X.size()
        
        X_norm = F.rms_norm(X, normalized_shape=(self.H,), weight=self.weight)

        qkv = self.W_total(X_norm)
        q, k, v = torch.split(qkv, H, dim=1)
        q = q.view(1, seq_len, self.num_heads, self.D)
        k = k.view(1, seq_len, self.num_heads, self.D)
        v = v.view(1, seq_len, self.num_heads, self.D)

        cache_seqlens = torch.tensor([self.context_length], dtype=torch.int32, device=X.device)
        
        attn_output = flash_attn_with_kvcache(
            q=q,
            k_cache=self.cache_K,
            v_cache=self.cache_V,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            softmax_scale=1.0 / (self.D ** 0.5),
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
        )

        # Reshape and project output
        attn_output = attn_output.contiguous().view(seq_len, self.H)
        return attn_output
    
class TensorRT_kernel(nn.Module):
    def __init__(self, M, H, D, num_heads, cache_K, cache_V, context_length):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.num_heads = num_heads
        self.context_length = context_length
        self.P = context_length  # Add P for cache length
        
        self.W_tot = nn.Linear(H, 3*H, bias=False)

        self.cache_K = cache_K
        self.cache_V = cache_V

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale
            
            def forward(self, q, k, v):
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                weights = F.softmax(scores, dim = -1)

                output = torch.matmul(weights, v)
                output = output.transpose(1, 2)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale
            model = AttentionModel(1.0 / math.sqrt(self.D))
            
            # Dummy inputs for export
            batch = 1
            seq_q = 16  # M
            seq_k = self.context_length + seq_q  # P + M
            
            dummy_q = torch.randn(batch, seq_q, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            dummy_k = torch.randn(batch, seq_k, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            dummy_v = torch.randn(batch, seq_k, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_q, dummy_k, dummy_v),
                onnx_path,
                input_names=['q', 'k', 'v'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        qkv = self.W_tot(X)
        q, k, v = torch.split(qkv, H, dim=1)

        q = q.view(1, seq_len, self.num_heads, self.D)
        k = k.view(1, seq_len, self.num_heads, self.D)
        v = v.view(1, seq_len, self.num_heads, self.D)

        k_wcache = torch.cat([self.cache_K[:, :self.P], k], dim=1)
        v_wcache = torch.cat([self.cache_V[:, :self.P], v], dim=1)

        output = torch.empty((1, seq_len, self.num_heads, self.D), dtype=torch.float16, device='cuda')

        bindings = [
            q.data_ptr(),
            k_wcache.data_ptr(),
            v_wcache.data_ptr(),
            output.data_ptr()
        ]

        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # Reshape output
        output = output.view(seq_len, self.H)
        
        return output

class TensorRT_kernel_RMSNorm(nn.Module):
    def __init__(self, M, H, D, num_heads, cache_K, cache_V, context_length):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.num_heads = num_heads
        self.context_length = context_length
        self.P = context_length  # Add P for cache length
        
        self.W_tot = nn.Linear(H, 3*H, bias=False)

        self.cache_K = cache_K
        self.cache_V = cache_V

        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(H))

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale
            
            def forward(self, q, k, v):
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                weights = F.softmax(scores, dim = -1)

                output = torch.matmul(weights, v)
                output = output.transpose(1, 2)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale
            model = AttentionModel(1.0 / math.sqrt(self.D))
            
            # Dummy inputs for export
            batch = 1
            seq_q = 16  # M
            seq_k = self.context_length + seq_q  # P + M
            
            dummy_q = torch.randn(batch, seq_q, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            dummy_k = torch.randn(batch, seq_k, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            dummy_v = torch.randn(batch, seq_k, self.num_heads, self.D, dtype=torch.float16, device='cuda')
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_q, dummy_k, dummy_v),
                onnx_path,
                input_names=['q', 'k', 'v'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        variance = X.pow(2).mean(-1, keepdim=True)
        X_normed = X * torch.rsqrt(variance + self.eps)
        X_normed = self.weight * X_normed

        qkv = self.W_tot(X_normed)
        q, k, v = torch.split(qkv, H, dim=1)

        q = q.view(1, seq_len, self.num_heads, self.D)
        k = k.view(1, seq_len, self.num_heads, self.D)
        v = v.view(1, seq_len, self.num_heads, self.D)

        k_wcache = torch.cat([self.cache_K[:, :self.P], k], dim=1)
        v_wcache = torch.cat([self.cache_V[:, :self.P], v], dim=1)

        output = torch.empty((1, seq_len, self.num_heads, self.D), dtype=torch.float16, device='cuda')

        bindings = [
            q.data_ptr(),
            k_wcache.data_ptr(),
            v_wcache.data_ptr(),
            output.data_ptr()
        ]

        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # Reshape output
        output = output.view(seq_len, self.H)
        
        return output

# M = 16
# H = 32
# D = 64
# P = 2048
# N = 2048

# ITER = 100

# X = torch.randn((M, N), device=device, dtype=dtype) * 0.1
# cache_K = torch.randn((1, P+M, H, D), device=device, dtype=dtype) * 0.1
# cache_V = torch.randn((1, P+M, H, D), device=device, dtype=dtype) * 0.1

# rms_block = TensorRT_kernel_RMSNorm(M, N, D, H, cache_K.clone(), cache_V.clone(), P).cuda()
# rms_block.half()
# with torch.no_grad():
#     for _ in range(10):
#         out = rms_block(X)
#     torch.cuda.synchronize()
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     start_event.record()
#     for _ in range(ITER):
#         out = rms_block(X)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
# print(f"{elapsed_time/ITER:.3f}ms")

# block = Torch_kernel1(N, D, H, cache_K.clone(), cache_V.clone(), P).cuda()
# block.half()
# with torch.no_grad():
#     for _ in range(10):
#         out = block(X)
#     torch.cuda.synchronize()
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()
#     for _ in range(ITER):
#         out = block(X)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
# print(f"{elapsed_time/ITER:.3f}ms")

# trt_block = TensorRT_kernel(M, N, D, H, cache_K.clone(), cache_V.clone(), P).cuda()
# trt_block.half()
# with torch.no_grad():
#     for _ in range(10):
#         out = trt_block(X)
#     torch.cuda.synchronize()

#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()
#     for _ in range(ITER):
#         out = trt_block(X)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
# print(f"{elapsed_time/ITER:.3f}ms")