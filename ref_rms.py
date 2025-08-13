import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
import torch
import flashinfer
import math
import tensorrt as trt
import tempfile
import os
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache

# TensorRT-LLM imports
try:
    import tensorrt_llm
    from tensorrt_llm import Tensor
    from tensorrt_llm.functional import gpt_attention, rms_norm, AttentionMaskType
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.runtime import Session, TensorInfo
    TRTLLM_AVAILABLE = True
except ImportError as e:
    print(f"TensorRT-LLM not available: {e}")
    TRTLLM_AVAILABLE = False

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

class Vanila(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # k_cache = torch.cat([cache_K[:, :self.P, :, :], k], dim=1)
        # v_cache = torch.cat([cache_V[:, :self.P, :, :], v], dim=1)
        
        # self.cache_K[:, self.P:self.P+self.M, :] = k
        # self.cache_V[:, self.P:self.P+self.M, :] = v
        
        # q = q.transpose(0, 1)

        # scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        
        # output = torch.matmul(weights, self.cache_V)
        # output = output.view(self.M, self.H * self.D)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output

class PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q)
        k = torch.matmul(X_norm, self.W_k)
        v = torch.matmul(X_norm, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output

class KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.noise = noise.to(device=device, dtype=dtype)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        # q = torch.matmul(X_norm, self.W_q)
        # k = torch.matmul(X_norm, self.W_k)
        # v = torch.matmul(X_norm, self.W_v)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output, perturb_out

class QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        # q = torch.matmul(X_norm, self.W_q)
        # k = torch.matmul(X_norm, self.W_k)
        # v = torch.matmul(X_norm, self.W_v)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output

class RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        # q = torch.matmul(X_norm, self.W_q)
        # k = torch.matmul(X_norm, self.W_k)
        # v = torch.matmul(X_norm, self.W_v)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output, weights_sum, weights_sqr_sum

class SimpleAttention(nn.Module):
    """Simple manual attention implementation for comparison"""
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        seq_len  = X.shape[0]
        
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(seq_len, self.H, self.D)
        k = k.view(seq_len, self.H, self.D)
        v = v.view(seq_len, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        self.cache_K[:, self.P:self.P+seq_len, :] = k
        self.cache_V[:, self.P:self.P+seq_len, :] = v

        q = q.transpose(0, 1)

        scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, self.cache_V)
        output = output.view(seq_len, self.H * self.D)

        return output

class TensorRT_Vanila(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P  # Add P for cache length

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_normed = X * torch.rsqrt(variance)
                # X_normed = self.weight * X_normed

                # QKV projection using separate matrices
                '''
                F.linear(X, W) = X @ W.T
                torch.matmul(X, W) = X @ W
                '''
                # q = F.linear(X_normed, self.W_q)
                # k = F.linear(X_normed, self.W_k)
                # v = F.linear(X_normed, self.W_v)

                q = torch.matmul(X, self.W_q)
                k = torch.matmul(X, self.W_k)
                v = torch.matmul(X, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :, :], v], dim=1)
                # cache_K[:, self.P:self.P+seq_len, :] = k
                # cache_V[:, self.P:self.P+seq_len, :] = v

                # scores = torch.matmul(q, cache_K.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                # output = torch.matmul(weights, cache_V)
                # output = output.view(seq_len, self.H*self.D)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
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

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_PreNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                q = torch.matmul(X_norm, self.W_q)
                k = torch.matmul(X_norm, self.W_k)
                v = torch.matmul(X_norm, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
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

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_KeyFormer(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, noise, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.noise = noise.to(device=device, dtype=dtype)
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, noise, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.noise = noise
                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_norm = X * torch.rsqrt(variance)

                # q = torch.matmul(X_norm, self.W_q)
                # k = torch.matmul(X_norm, self.W_k)
                # v = torch.matmul(X_norm, self.W_v)


                q = torch.matmul(X, self.W_q)
                k = torch.matmul(X, self.W_k)
                v = torch.matmul(X, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                perturb = (scores+self.noise) / 1.5
                weights = F.softmax(scores, dim=-1)
                perturb_out = F.softmax(perturb, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output, perturb_out

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.noise,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'perturb_out'],
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

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        perturb_out = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            perturb_out.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_QKNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_norm = X * torch.rsqrt(variance)

                # q = torch.matmul(X_norm, self.W_q)
                # k = torch.matmul(X_norm, self.W_k)
                # v = torch.matmul(X_norm, self.W_v)

                q = torch.matmul(X, self.W_q)
                k = torch.matmul(X, self.W_k)
                v = torch.matmul(X, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                q_var = q.pow(2).mean(-1, keepdim=True)
                k_var = k.pow(2).mean(-1, keepdim=True)
                q_norm = q * torch.rsqrt(q_var)
                k_norm = k * torch.rsqrt(k_var)

                k_cache = torch.cat([cache_K[:, :self.P, :], k_norm], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
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

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_RoCo(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_norm = X * torch.rsqrt(variance)

                # q = torch.matmul(X_norm, self.W_q)
                # k = torch.matmul(X_norm, self.W_k)
                # v = torch.matmul(X_norm, self.W_v)

                q = torch.matmul(X, self.W_q)
                k = torch.matmul(X, self.W_k)
                v = torch.matmul(X, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                weights_sum = weights.sum(dim=1)
                weights_sqr_sum = weights.pow(2).sum(dim=1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output, weights_sum, weights_sqr_sum

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'weights_sum', 'weights_sqr_sum'],
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

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        weights_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        weights_sqr_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            weights_sum.data_ptr(),
            weights_sqr_sum.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class FlashInfer_Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # QKV projection
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE"
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE"
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.noise = noise

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)
        
        # # QKV projection
        # q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        # k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        # v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE"
        )
        
        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)
        
        # QKV projection
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q_norm,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE"
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)
        
        # QKV projection
        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE"
        )

        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)

        return output.squeeze(0).view(self.M, self.N)

class SimpleBatchAttention(nn.Module):
    """Simple manual attention implementation for comparison"""
    def __init__(self, B, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        seq_len  = X.shape[1]
        
        """Reference implementation with batch dimension using tensor operations"""
        # Step 1: RMS normalization for all batches at once
        variance = X.pow(2).mean(-1, keepdim=True)  # (B, M, 1)
        X_norm = X * torch.rsqrt(variance)  # (B, M, N)
        
        # Step 2: QKV projection for all batches and heads
        # X_norm: (B, M, N), WQ/WK/WV: (B, N, N)
        Q_all = torch.matmul(X_norm, self.W_q)  # (B, M, N)
        K_all = torch.matmul(X_norm, self.W_k)  # (B, M, N)
        V_all = torch.matmul(X_norm, self.W_v)  # (B, M, N)
        
        # Step 3: Reshape to separate heads
        # (B, M, N) -> (B, M, H, D) -> (B, H, M, D)
        Q = Q_all.view(self.B, seq_len, self.H, self.D).permute(0, 2, 1, 3)  # (B, H, M, D)
        K = K_all.view(self.B, seq_len, self.H, self.D).permute(0, 2, 1, 3)  # (B, H, M, D)
        V = V_all.view(self.B, seq_len, self.H, self.D).permute(0, 2, 1, 3)  # (B, H, M, D)
        
        # Step 4: Update K/V cache
        self.cache_K[:, :, self.P:self.P+seq_len, :] = K
        self.cache_V[:, :, self.P:self.P+seq_len, :] = V
        
        # Step 5: Attention computation for all batches and heads
        # Q: (B, H, M, D), K_cache: (B, H, P+M, D)
        # Transpose K_cache for matmul: (B, H, D, P+M)
        K_T = self.cache_K.transpose(-2, -1)  # (B, H, D, P+M)
        scores = torch.matmul(Q, K_T)  # (B, H, M, P+M)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, M, P+M)
        
        # Apply attention to values
        # attn_weights: (B, H, M, P+M), V_cache: (B, H, P+M, D)
        O = torch.matmul(attn_weights, self.cache_V)  # (B, H, M, D)
        
        # Step 6: Reshape output
        # (B, H, M, D) -> (B, M, H, D) -> (B, M, N)
        O2 = O.permute(0, 2, 1, 3).contiguous().view(self.B, seq_len, self.N)  # (B, M, N)
        
        return O2

class TensorRT_Batch_RMSNorm(nn.Module):
    def __init__(self, B, M, H, D, num_heads, cache_K, cache_V, context_length, W_q, W_k, W_v):
        super().__init__()
        self.B = B
        self.M = M
        self.H = H
        self.D = D
        self.num_heads = num_heads
        self.context_length = context_length
        self.P = context_length  # Add P for cache length

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, B, H, D, num_heads, P, W_q, W_k, W_v):
                super().__init__()
                self.B = B
                self.H = H
                self.D = D
                self.num_heads = num_heads
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                self.register_buffer('W_q', W_q)
                self.register_buffer('W_k', W_k)
                self.register_buffer('W_v', W_v)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[1]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_normed = X * torch.rsqrt(variance)
                # X_normed = self.weight * X_normed

                # QKV projection using separate matrices
                '''
                F.linear(X, W) = X @ W.T
                torch.matmul(X, W) = X @ W
                '''
                # q = F.linear(X_normed, self.W_q)
                # k = F.linear(X_normed, self.W_k)
                # v = F.linear(X_normed, self.W_v)

                q = torch.matmul(X_normed, self.W_q)
                k = torch.matmul(X_normed, self.W_k)
                v = torch.matmul(X_normed, self.W_v)

                q = q.view(self.B, seq_len, self.num_heads, self.D).permute(0, 2, 1, 3)
                k = k.view(self.B, seq_len, self.num_heads, self.D).permute(0, 2, 1, 3)
                v = v.view(self.B, seq_len, self.num_heads, self.D).permute(0, 2, 1, 3)

                # Update cache
                # k_cache = torch.cat([cache_K[:, :self.P, :, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :, :], v], dim=1)
                cache_K[:, :, self.P:self.P+seq_len, :] = k
                cache_V[:, :, self.P:self.P+seq_len, :] = v

                # Transpose for attention
                # q = q.transpose(1, 2)
                # k_cache = k_cache.transpose(1, 2)
                # v_cache = v_cache.transpose(1, 2)

                # Attention
                scores = torch.matmul(q, cache_K.transpose(-2, -1))
                weights = F.softmax(scores, dim=-1)

                output = torch.matmul(weights, cache_V)
                output = output.permute(0, 2, 1, 3).contiguous().view(self.B, seq_len, self.num_heads*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.B,
                self.H,
                self.D,
                self.num_heads,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(self.B, seq_len, self.H, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
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
        B, seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(B, seq_len, self.H, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

# class Torch_kernel1(nn.Module):
#     def __init__(self, H, D, num_heads, cache_K, cache_V, context_length, W_q=None, W_k=None, W_v=None):
#         super(Torch_kernel1, self).__init__()
#         self.H = H
#         self.D = D
#         self.num_heads = num_heads

#         self.weight = nn.Parameter(torch.ones(self.H))
#         if W_q is not None and W_k is not None and W_v is not None:
#             # Use provided weights
#             self.W_q = nn.Parameter(W_q.clone())
#             self.W_k = nn.Parameter(W_k.clone())
#             self.W_v = nn.Parameter(W_v.clone())
#         else:
#             # Create random weights
#             self.W_total = nn.Linear(self.N, 3*self.N, bias=False)

#         self.cache_K = cache_K 
#         self.cache_V = cache_V

#         self.context_length = context_length

#     def forward(self, X):
#         seq_len, H = X.size()
        
#         X_norm = F.rms_norm(X, normalized_shape=(self.H,), weight=self.weight)

#         if hasattr(self, 'W_q'):
#             # Use separate projection matrices
#             q = F.linear(X_norm, self.W_q)
#             k = F.linear(X_norm, self.W_k)
#             v = F.linear(X_norm, self.W_v)
#         else:
#             # Use combined projection matrix
#             qkv = self.W_total(X_norm)
#             q, k, v = torch.split(qkv, self.N, dim=1)

#         q = q.view(1, seq_len, self.num_heads, self.D)
#         k = k.view(1, seq_len, self.num_heads, self.D)
#         v = v.view(1, seq_len, self.num_heads, self.D)

#         cache_seqlens = torch.tensor([self.context_length], dtype=torch.int32, device=X.device)
        
#         attn_output = flash_attn_with_kvcache(
#             q=q,
#             k_cache=self.cache_K,
#             v_cache=self.cache_V,
#             k=k,
#             v=v,
#             cache_seqlens=cache_seqlens,
#             softmax_scale=1.0 / (self.D ** 0.5),
#             causal=False,
#             window_size=(-1, -1),
#             alibi_slopes=None,
#         )

#         # Reshape and project output
#         attn_output = attn_output.contiguous().view(seq_len, self.H)
#         return attn_output

# M = 16
# H = 32
# D = 64
# P = 2048
# N = 2048

# M = 16
# H = 71
# D = 64
# P = 512
# N = 4544

# ITER = 100

# X = torch.randn((M, N), device=device, dtype=dtype) * 0.1
# # Cache should initially contain P elements, new K/V will be appended
# num_heads = N // D
# cache_K = torch.randn((1, P, num_heads, D), device=device, dtype=dtype) * 0.1
# cache_V = torch.randn((1, P, num_heads, D), device=device, dtype=dtype) * 0.1

# print("TensorRT RMS Norm")
# # Pass N as hidden dimension and num_heads as the number of attention heads
# rms_block = TensorRT_kernel_RMSNorm(M, N, D, num_heads, cache_K.clone(), cache_V.clone(), P).to(device)
# rms_block.half()
# with torch.no_grad():
#     for _ in range(10):
#         out = rms_block(X)
#     torch.cuda.synchronize()
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
    
#     torch.cuda.nvtx.range_push("kernel_section")
#     start_event.record()
#     for _ in range(ITER):
#         out = rms_block(X)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
# print(f"{elapsed_time/ITER:.3f}ms")

# print("Flash Decoding")
# block = Torch_kernel1(N, D, H, cache_K.clone(), cache_V.clone(), P).to(device)
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

# print("Simple Attention")
# trt_block = SimpleAttention(M, N, D, P, cache_K.squeeze(0).clone(), cache_V.squeeze(0).clone()).to(device)
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