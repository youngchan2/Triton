from convert_module import convert_ir_to_triton
from ref_rms import Vanila, PreNorm, KeyFormer, QKNorm, RoCo, TensorRT_Vanila, TensorRT_PreNorm, TensorRT_KeyFormer, TensorRT_QKNorm, TensorRT_RoCo, FlashInfer_Vanilla, FlashInfer_PreNorm, FlashInfer_KeyFormer, FlashInfer_QKNorm, FlashInfer_RoCo
from baseline.inductor import benchmark_rms
import argparse
import torch
import torch.nn.functional as F
import math
import importlib.util
import sys

parser = argparse.ArgumentParser(description="Convert Attacc IR to Triton kernel")
parser.add_argument("--n", type=int, default=0, help="Case number to convert")
parser.add_argument("--m", type=str, default="llama", help="Input model type")
parser.add_argument("--t", type=str, default="vanilla", help="RMS Type")
parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
args = parser.parse_args()

num = args.n
option = args.o
model = args.m
rms = args.t
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

case_file = f"./evaluation/{rms}/{rms}_{model}_case{num}.txt"
output_file = f"./evaluation/{rms}/{rms}_{model}_benchmark{num}.py"
module_name = f"{rms}_{model}_best"

with open(case_file, "r") as f:
    llama_ir = f.read().strip()

if model == "falcon":
    M = 16
    D = 64
    N = 4544
    P = 1024
    H = 71
    
    constants = {
        'M': 16,
        'D': 64,
        'N': 4544,
        'P': 1024,
        'H': 71
    }
elif model == "llama":
    M = 16
    D = 128
    N = 4096
    H = 32
    P = 1024
    
    constants = {
        'M': 16,
        'D': 128,
        'N': 4096,
        'P': 1024,
        'H': 32
    }

tensor_shapes = {
            'X': ('M', 'N'),
            'X2': ('M', ),
            'X_norm': ('M', 'N'),
            
            'WQ': ('N', 'N'),
            'WK': ('N', 'N'),
            'WV': ('N', 'N'),

            'Q': ('H', 'M', 'D'),
            'K': ('H', 'M', 'D'),
            'V': ('H', 'M', 'D'),

            'Q1': ('M', 'N'),
            'K1': ('M', 'N'),
            'V1': ('M', 'N'),

            'Q2': ('M', 'H', 'D'),
            'K2': ('M', 'H', 'D'),
            'V2': ('M', 'H', 'D'),

            'K_cache': ('H', 'P+M', 'D'),
            'V_cache': ('H', 'P+M', 'D'),

            'K': ('H', 'M', 'D'),
            'V': ('H', 'M', 'D'),

            'O': ('H', 'M', 'D'),
            'O1': ('M', 'H', 'D'),
            'O2': ('M', 'N'),

            'C': ('H', 'M', 'P'),
            'C_exp': ('H', 'M', 'P+M'),
            'C_div': ('H', 'M', 'P+M'),
            'C_sum': ('H', 'M'),
            'noise': ('H', 'M', 'P+M'),
            'C_perturb': ('H', 'M', 'P+M'),
            'C_exp_perturb': ('H', 'M', 'P+M'),
            'C_sum_perturb': ('H', 'M', 'P+M'),
            'C_div_perturb': ('H', 'M', 'P+M'),
            'C_out': ('H', 'P+M'),
            'C_out1': ('H', 'P+M'),
            'C_out2': ('H', 'P+M'),

            'Q_norm': ('H', 'M', 'D'),
            'K_norm': ('H', 'M', 'D')
        }

# Convert IR to Triton kernel
def start_conversion():

    triton_code = convert_ir_to_triton(llama_ir, tensor_shapes, constants)

    # Save the generated kernel
    with open(output_file, "w") as f:
        f.write(triton_code)

    print("=" * 50)
    print("✓ Triton kernel generated successfully!")

def reference_kernel(X, WQ, WK, WV, K_cache, V_cache, M, N, D, H, P):
    """Reference implementation matching rms_benchmark0.py exactly"""
    O2 = torch.zeros((M, N), device=X.device, dtype=X.dtype)
    
    # Step 1: RMS normalization (without eps, matching TensorRT)
    variance = X.pow(2).mean(-1, keepdim=True)  # (M, 1)
    X_norm = X * torch.rsqrt(variance)  # (M, N)
    
    # Process each head (block) independently
    for head_idx in range(H):
        n_start = head_idx * D
        n_end = n_start + D
        
        # Step 2: QKV projection for this head's dimensions
        Q1 = torch.matmul(X_norm, WQ[:, n_start:n_end])  # (M, D)
        K1 = torch.matmul(X_norm, WK[:, n_start:n_end])  # (M, D)
        V1 = torch.matmul(X_norm, WV[:, n_start:n_end])  # (M, D)
        
        # Step 3: Reshape to add head dimension
        Q = Q1.unsqueeze(0)  # (1, M, D)
        K = K1.unsqueeze(0)  # (1, M, D)
        V = V1.unsqueeze(0)  # (1, M, D)
        
        # Step 4: Update K/V cache for this head
        K_cache[head_idx, P:P+M, :] = K.squeeze(0)
        V_cache[head_idx, P:P+M, :] = V.squeeze(0)
        
        # Step 5: Attention computation
        # Compute attention scores for all positions
        scores = torch.matmul(Q, K_cache[head_idx, :P+M, :].T)  # (1, M, P+M)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (1, M, P+M)
        
        # Apply attention to values
        O = torch.matmul(attn_weights, V_cache[head_idx, :P+M, :])  # (1, M, D)
        
        # Step 6: Store output for this head
        O2[:, n_start:n_end] = O.squeeze(0)  # (M, D)
    
    return O2

def start_test():
    BLOCK_N = 64
    BLOCK_K = 32
    BLOCK_P = 16

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Use much smaller values to avoid numerical overflow
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype).clamp(-1,1) * std
    X2 = torch.randn(M, device=device, dtype=dtype)
    X_norm = torch.zeros((M, N), device=device, dtype=dtype)
    WQ = torch.randn((N, N), device=device, dtype=dtype).clamp(-1,1) * std
    WK = torch.randn((N, N), device=device, dtype=dtype).clamp(-1,1) * std
    WV = torch.randn((N, N), device=device, dtype=dtype).clamp(-1,1) * std
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype).clamp(-1,1) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype).clamp(-1,1) * std
    C = torch.zeros((H, M, P), device=device, dtype=dtype)
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_sum = torch.zeros((H, M), device=device, dtype=dtype)
    K = torch.zeros((H, M, D), device=device, dtype=dtype)
    K1 = torch.zeros((M, D), device=device, dtype=dtype)
    K2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    O = torch.zeros((H, M, D), device=device, dtype=dtype)
    O1 = torch.zeros((H, M, D), device=device, dtype=dtype)
    O2 = torch.zeros((M, N), device=device, dtype=dtype)
    Q = torch.zeros((H, M, D), device=device, dtype=dtype)
    Q1 = torch.zeros((M, D), device=device, dtype=dtype)
    Q2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    V = torch.zeros((H, M, D), device=device, dtype=dtype)
    V1 = torch.zeros((M, D), device=device, dtype=dtype)
    V2 = torch.zeros((M, H, D), device=device, dtype=dtype)

    # Key Former
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype)
    C_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_sum_perturb = torch.zeros((H, M), device=device, dtype=dtype)
    C_div_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)

    # QKNorm
    Q_norm = torch.zeros((H, M, D), device=device, dtype=dtype)
    K_norm = torch.zeros((H, M, D), device=device, dtype=dtype)

    # RoCo
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype)
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype)

    out = O2.clone()
    ITER = 100

    print("=" * 50)
    print("Starting kernel execution...")

    spec = importlib.util.spec_from_file_location(module_name, output_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")

    tensor_params = getattr(module, 'TENSOR_PARAMS')
    block_params = getattr(module, 'BLOCK_PARAMS')

    tensors = {
        'X': X,
        'X2': X2,
        'X_norm': X_norm,
        'WQ': WQ,
        'WK': WK,
        'WV': WV,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'C': C,
        'C_exp': C_exp,
        'C_sum': C_sum,
        'K': K,
        'K1': K1,
        'K2': K2,
        'O': O,
        'O1': O1,
        'O2': O2,
        'Q': Q,
        'Q1': Q1,
        'Q2': Q2,
        'V': V,
        'V1': V1,
        'V2': V2,

        'noise': noise,
        'C_perturb': C_perturb,
        'C_exp_perturb': C_exp_perturb,
        'C_sum_perturb': C_sum_perturb,
        'C_div_perturb': C_div_perturb,
        'C_out': C_out,

        'Q_norm': Q_norm,
        'K_norm': K_norm,

        'C_out1': C_out1,
        'C_out2': C_out2
    }

    blocks = {
        'block_k': BLOCK_K,
        'block_p': BLOCK_P
    }

    args = []
    for param in tensor_params:
        if param in tensors:
            args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in block_params:
        if param in blocks:
            args.append(blocks[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")

    # ------------------- Tile -------------------
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        for _ in range(10):
            forward(*args)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            forward(*args)
    stream.synchronize()

    start_rms = torch.cuda.Event(enable_timing=True)
    end_rms = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_rms.record()
        for _ in range(100):
            graph.replay()
        end_rms.record()  
    stream.synchronize()      

    time = start_rms.elapsed_time(end_rms) / 100

    print(f"Kernel execution completed!: {time}ms")
    print(O2)

    match rms:
        case "vanilla":
            trt = TensorRT_Vanila(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = Vanila(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "prenorm":
            trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "keyformer":
            trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
            ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "qknorm":
            trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "roco":
            trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)

    print("=" * 50)
    print("Starting ref kernel execution...")
    print(f"\nTesting {rms.upper()} Flash Infer...")
    print("\nTesting Correct Flash Infer...")
    # ------------------- Flash Infer -------------------
    fi.half()
    with torch.no_grad():
        for _ in range(10):
            out = fi(X)
        torch.cuda.synchronize()

        start_fi = torch.cuda.Event(enable_timing=True)
        end_fi = torch.cuda.Event(enable_timing=True)

        start_fi.record()
        for _ in range(ITER):
            out = fi(X)
        end_fi.record()
        torch.cuda.synchronize()
        fi_time = start_fi.elapsed_time(end_fi) / ITER
        print(f"FI: {fi_time}ms")
    print(out)


    print("=" * 50)
    print(f"\nTesting {rms.upper()} TensorRT...")
    print("\nTesting Correct TensorRT...")
    # ------------------- Tensor RT -------------------
    
    trt.half()
    with torch.no_grad():
        for _ in range(10):
            out = trt(X)
        torch.cuda.synchronize()

        start_rt = torch.cuda.Event(enable_timing=True)
        end_rt = torch.cuda.Event(enable_timing=True)

        start_rt.record()
        for _ in range(ITER):
            out = trt(X)
        end_rt.record()
        torch.cuda.synchronize()
        rt_time = start_rt.elapsed_time(end_rt) / ITER
        print(f"TRT: {rt_time}ms")
    print(out)

    # ------------------- Torch Inductor -------------------
    print("\nTesting Torch Inductor Implementation...")

    benchmark_rms(ti.eval(), X)
    
    # print("\nComparing results...")
    # if torch.allclose(O2, out, rtol=1e-3, atol=1e-4):
    #     print("✓ Results match!")
    # else:
    #     print("✗ Results do not match!")
    #     max_diff = torch.abs(O2 - out).max()
    #     print(f"Maximum difference: {max_diff}")
    # print("=" * 50)

print(f"[Case{num}]")
if option == 0:
    start_conversion()
elif option == 1:
    start_test()
else:
    start_conversion()
    start_test()