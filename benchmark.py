import torch
import triton
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
import importlib.util
import sys
import os
import traceback
from dataclasses import dataclass
import json

from IrParser import IRParser
from TritonGen import TritonCodeGen


@dataclass
class BenchmarkResult:
    ir_id: int
    ir_expression: str
    execution_time: float
    error: Optional[str] = None
    
    
class LoRABenchmark:
    def __init__(self, batch_size=1, seq_len=32, hidden_dim=64, rank=16):
        """Initialize benchmark with typical LoRA dimensions."""
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # Define tensor shapes for LoRA computation
        # Note: Using same dimensions as convert_case1.py
        M = batch_size * seq_len  # 32
        N = hidden_dim            # 64
        P = hidden_dim            # 64
        R = rank                  # 16
        
        self.tensor_shapes = {
            # Primary tensors
            'X': (M, N),  # Input activations (32x64)
            'W': (N, P),  # Original weight matrix (64x64)
            'A': (N, R),  # LoRA matrix A (64x16)
            'B': (R, P),  # LoRA matrix B (16x64)
            'O': (M, P),  # Output (32x64)
            
            # Intermediate tensors (for seq operations)
            'C': (M, P),  # Intermediate result, same shape as O
            'D': (M, R),  # Intermediate result: X @ A = (M, N) @ (N, R) = (M, R)
            'E': (M, P),  # Intermediate result, same shape as O
        }
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("CUDA device not available. Triton requires CUDA.")
            
        # Create test tensors
        self.create_test_tensors()
        
        # Track temp files for cleanup
        self._temp_files = []
        
    def create_test_tensors(self):
        """Create random test tensors for benchmarking."""
        self.tensors = {}
        for name, shape in self.tensor_shapes.items():
            if name == 'O':  # Output tensor
                continue
            elif name in ['C', 'D', 'E']:  # Intermediate tensors
                # Initialize to zero since they're used as accumulators
                self.tensors[name] = torch.zeros(shape, dtype=torch.float32, device=self.device)
            else:  # Input tensors
                self.tensors[name] = torch.randn(shape, dtype=torch.float32, device=self.device)
    
    def parse_ir_file(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse the IR expressions file and extract all expressions."""
        expressions = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    # Extract IR ID and expression
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        id_part = parts[0]
                        try:
                            ir_id = int(id_part)
                            ir_expr = parts[1].strip()
                            expressions.append((ir_id, ir_expr))
                        except:
                            continue
                            
        return expressions
    
    def generate_kernel_code(self, ir_expr: str) -> Optional[str]:
        """Generate Triton kernel code from IR expression."""
        try:
            # Parse IR
            parser = IRParser()
            ast = parser.parse(ir_expr)
            
            if ast is None:
                return None
                
            # Generate Triton code
            codegen = TritonCodeGen()
            kernel_code = codegen.generate(ast, self.tensor_shapes)
            
            return kernel_code
            
        except Exception as e:
            print(f"Error generating kernel: {e}")
            traceback.print_exc()
            return None
    
    def compile_and_load_kernel(self, kernel_code: str, kernel_id: int) -> Optional[callable]:
        """Compile Triton kernel code and return callable function."""
        try:
            # Create temporary module file
            module_name = f"lora_kernel_{kernel_id}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(module_name, temp_file)
            module = importlib.util.module_from_spec(spec)
            
            # Keep the file path in the module for Triton to find source
            module.__file__ = temp_file
            
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Don't delete temp file immediately - Triton needs it
            # Store temp file path for later cleanup
            self._temp_files = getattr(self, '_temp_files', [])
            self._temp_files.append(temp_file)
            
            # Get the kernel function
            # For seq cases, look for forward function first
            if hasattr(module, 'forward'):
                return getattr(module, 'forward')
            
            # Otherwise look for single kernel function
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and 'kernel' in attr_name.lower() and not attr_name.startswith('_'):
                    return attr
                    
            return None
            
        except Exception as e:
            print(f"Error compiling kernel: {e}")
            traceback.print_exc()
            if 'temp_file' in locals() and os.path.exists(temp_file):
                # Print the generated code for debugging
                with open(temp_file, 'r') as f:
                    print("Generated kernel code:")
                    print(f.read())
                os.unlink(temp_file)
            return None
    
    def benchmark_kernel(self, kernel_fn: callable, warmup_runs: int = 10, benchmark_runs: int = 100) -> float:
        """Benchmark a single kernel and return execution time in milliseconds."""
        try:
            # Check if this is a forward function (for seq cases) or a kernel
            is_forward_fn = kernel_fn.__name__ == 'forward'
            
            # Create output tensor
            output = torch.zeros(self.tensor_shapes['O'], dtype=torch.float32, device=self.device)
            
            # Prepare arguments
            args = []
            
            if is_forward_fn:
                # For forward functions, check the actual signature
                import inspect
                sig = inspect.signature(kernel_fn)
                param_names = list(sig.parameters.keys())
                
                # Add tensors based on the actual function signature
                for name in param_names:
                    name_upper = name.upper()
                    if name_upper == 'O':
                        args.append(output)
                    elif name_upper in self.tensors:
                        args.append(self.tensors[name_upper])
                    else:
                        # Try lowercase as well
                        if name in self.tensors:
                            args.append(self.tensors[name])
                        elif name_upper in self.tensor_shapes:
                            # Create tensor if it doesn't exist but is in shapes
                            if name_upper not in self.tensors:
                                self.tensors[name_upper] = torch.zeros(
                                    self.tensor_shapes[name_upper], 
                                    dtype=torch.float32, 
                                    device=self.device
                                )
                            args.append(self.tensors[name_upper])
            else:
                # For kernels, use the original logic
                # Add all tensors in alphabetical order
                for name in sorted(['X', 'W', 'A', 'B', 'C', 'D', 'E', 'O']):
                    if name == 'O':
                        # Output tensor
                        args.append(output)
                    elif name in self.tensors:
                        args.append(self.tensors[name])
            
            if is_forward_fn:
                # Forward functions are called directly without grid
                # Warmup runs
                for _ in range(warmup_runs):
                    kernel_fn(*args)
                
                # Synchronize before timing
                torch.cuda.synchronize()
                
                # Benchmark runs
                start_time = time.time()
                for _ in range(benchmark_runs):
                    kernel_fn(*args)
                torch.cuda.synchronize()
                end_time = time.time()
            else:
                # Single kernel with grid launch
                # Build arguments based on kernel signature
                kernel_args = []
                
                # Extract tensor names from kernel signature
                import inspect
                try:
                    sig = inspect.signature(kernel_fn.fn)
                except:
                    # For dynamically compiled kernels, use the kernel function directly
                    sig = inspect.signature(kernel_fn)
                param_names = list(sig.parameters.keys())
                
                # Build tensor pointer and shape/stride arguments
                tensor_ptrs_added = set()
                for param in param_names:
                    if param.endswith('_ptr'):
                        tensor_name = param[:-4]  # Remove '_ptr'
                        if tensor_name in self.tensors:
                            tensor = self.tensors[tensor_name]
                            kernel_args.append(tensor)
                            tensor_ptrs_added.add(tensor_name)
                        elif tensor_name == 'O':  # Output tensor
                            kernel_args.append(output)
                            tensor_ptrs_added.add(tensor_name)
                        elif tensor_name in ['C', 'D', 'E']:  # Intermediate tensors that might not be initialized
                            # Create them if needed
                            if tensor_name not in self.tensors:
                                self.tensors[tensor_name] = torch.zeros(
                                    self.tensor_shapes[tensor_name], 
                                    dtype=torch.float32, 
                                    device=self.device
                                )
                            kernel_args.append(self.tensors[tensor_name])
                            tensor_ptrs_added.add(tensor_name)
                    elif '_dim' in param:
                        # Shape parameter
                        tensor_name = param.split('_dim')[0]
                        dim_idx = int(param.split('_dim')[1])
                        if tensor_name in self.tensor_shapes:
                            kernel_args.append(self.tensor_shapes[tensor_name][dim_idx])
                        else:
                            kernel_args.append(64)  # Default dimension (adjusted for smaller tensors)
                    elif '_stride' in param:
                        # Stride parameter
                        tensor_name = param.split('_stride')[0]
                        stride_idx = int(param.split('_stride')[1])
                        if tensor_name in self.tensors:
                            kernel_args.append(self.tensors[tensor_name].stride()[stride_idx])
                        elif tensor_name == 'O':
                            kernel_args.append(output.stride()[stride_idx])
                        elif tensor_name in ['C', 'D', 'E'] and tensor_name in self.tensors:
                            kernel_args.append(self.tensors[tensor_name].stride()[stride_idx])
                        else:
                            # Default stride (adjusted for smaller tensors)
                            kernel_args.append(1 if stride_idx == 1 else 64)
                    elif param.startswith('BLOCK_'):
                        # Block size parameter - use smaller size to avoid shared memory issues
                        kernel_args.append(16)  # Changed from 32 to 16
                
                # Determine grid size based on output dimensions
                # For most kernels, the output is O, but check tensor shapes
                if 'O' in self.tensor_shapes:
                    M, P = self.tensor_shapes['O']
                else:
                    # Fallback to X dimensions
                    M, N = self.tensor_shapes['X']
                    P = N
                
                # Grid should cover the output space
                # Typically we parallelize over the P dimension (columns of output)
                grid = lambda meta: (triton.cdiv(P, 16),)  # Single dimension grid, matching BLOCK size
                # Warmup runs
                for _ in range(warmup_runs):
                    kernel_fn[grid](*kernel_args)
                
                # Synchronize before timing
                torch.cuda.synchronize()
                
                # Benchmark runs
                start_time = time.time()
                for _ in range(benchmark_runs):
                    kernel_fn[grid](*kernel_args)
                torch.cuda.synchronize()
                end_time = time.time()
            
            # Return average time in milliseconds
            avg_time = (end_time - start_time) / benchmark_runs * 1000
            return avg_time
            
        except Exception as e:
            print(f"Error benchmarking kernel: {e}")
            traceback.print_exc()
            # Re-raise to let the caller handle it
            raise
    
    def run_single_benchmark(self, ir_id: int, ir_expr: str) -> BenchmarkResult:
        """Run benchmark for a single IR expression."""
        try:
            # Generate kernel code
            kernel_code = self.generate_kernel_code(ir_expr)
            if kernel_code is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), "Failed to generate kernel")
            
            # Compile kernel
            kernel_fn = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_fn is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), "Failed to compile kernel")
            
            # Benchmark kernel
            exec_time = self.benchmark_kernel(kernel_fn)
            
            # Clean up GPU memory after each benchmark
            self.cleanup_gpu()
            
            # Also clean up the loaded module to prevent memory leaks
            module_name = f"lora_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return BenchmarkResult(ir_id, ir_expr, exec_time)
            
        except Exception as e:
            # Clean up GPU memory even on error
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float('inf'), str(e))
    
    def run_all_benchmarks(self, ir_file: str, max_expressions: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmarks for all IR expressions in the file."""
        # Parse IR expressions
        expressions = self.parse_ir_file(ir_file)
        min_num = 8650
        if max_expressions:
            expressions = expressions[min_num:min_num+max_expressions]
        
        print(f"Found {len(expressions)} IR expressions to benchmark")
        
        results = []
        for i, (ir_id, ir_expr) in enumerate(expressions):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(expressions)} expressions benchmarked")
            
            result = self.run_single_benchmark(ir_id, ir_expr)
            results.append(result)
            
            # Print immediate feedback for errors
            if result.error:
                print(f"  IR {ir_id}: Error - {result.error[:100]}...")
        
        return results
    
    def find_best_kernels(self, results: List[BenchmarkResult], top_k: int = 10) -> List[BenchmarkResult]:
        """Find the top-k fastest kernels."""
        # Filter out failed kernels
        valid_results = [r for r in results if r.error is None and r.execution_time != float('inf')]
        
        # Sort by execution time
        valid_results.sort(key=lambda x: x.execution_time)
        
        return valid_results[:top_k]
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to a JSON file."""
        data = []
        for r in results:
            data.append({
                'ir_id': r.ir_id,
                'ir_expression': r.ir_expression,
                'execution_time_ms': r.execution_time,
                'error': r.error
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def cleanup_gpu(self):
        """Clean up GPU memory and reset CUDA context."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                # Force synchronization
                torch.cuda.synchronize()
                
                # Clear all allocated tensors
                self.tensors.clear()
                
                # Multiple empty_cache calls to ensure thorough cleanup
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Small delay to ensure GPU cleanup
                time.sleep(0.1)
                
                # Recreate test tensors to ensure clean state
                self.create_test_tensors()
            
        except Exception as e:
            print(f"Warning: GPU cleanup failed: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self._temp_files = []


def main():
    """Main function to run LoRA IR benchmarks."""
    # Configuration
    IR_FILE = "/home/chani227/triton/part4/expressions/lora_skip_ft.txt"
    OUTPUT_FILE = "/home/chani227/triton/part4/benchmark_results.json"
    MAX_EXPRESSIONS = 50  # Set to small number for testing, None for all
    TOP_K = 10  # Number of best kernels to report
    
    # Initialize benchmark
    print("Initializing LoRA benchmark...")
    try:
        benchmark = LoRABenchmark(
            batch_size=1,
            seq_len=32,       # Changed to match convert_case1.py
            hidden_dim=64,    # Changed to match convert_case1.py
            rank=16
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    # Run benchmarks
    print(f"\nRunning benchmarks on IR expressions from: {IR_FILE}")
    results = benchmark.run_all_benchmarks(IR_FILE, max_expressions=MAX_EXPRESSIONS)
    
    # Save all results
    benchmark.save_results(results, OUTPUT_FILE)
    print(f"\nSaved all results to: {OUTPUT_FILE}")
    
    # Find and report best kernels
    best_kernels = benchmark.find_best_kernels(results, top_k=TOP_K)
    
    print(f"\nTop {TOP_K} fastest kernels:")
    print("-" * 80)
    for i, result in enumerate(best_kernels):
        print(f"{i+1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
        print(f"   Expression: {result.ir_expression[:100]}...")
        print()
    
    if best_kernels:
        print(f"\nFastest kernel: IR {best_kernels[0].ir_id} with {best_kernels[0].execution_time:.4f} ms")
        print(f"Expression: {best_kernels[0].ir_expression}")
    
    # Clean up temporary files
    benchmark.cleanup()


if __name__ == "__main__":
    main()