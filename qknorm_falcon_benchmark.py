import torch
import triton
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
import importlib.util
import sys
import os
import shutil
import traceback
from dataclasses import dataclass
from tqdm import tqdm
import json
import argparse

from convert_module import convert_ir_to_triton
# from ref_rms import TensorRT_kernel_RMSNorm,  SimpleAttention

@dataclass
class BenchmarkResult:
    ir_id: int
    ir_expression: str
    execution_time: float
    tensor_config: Dict[str, int]
    error: Optional[str] = None
    
    
class RMSBenchmark:
    def __init__(self, tensor_config: Dict[str, int]):
        """Initialize benchmark with given tensor configuration."""
        # Extract dimensions from config
        self.M = tensor_config['M']
        self.N = tensor_config['N']
        self.D = tensor_config['D']
        self.H = tensor_config['H']
        self.P = tensor_config['P']
        
        # Store the config
        self.tensor_config = tensor_config
        
        self.tensor_shapes = {
            'X': (self.M, self.N),
            'X2': (self.M, ),
            'X_norm': (self.M, self.N),
            
            'WQ': (self.N, self.N),
            'WK': (self.N, self.N),
            'WV': (self.N, self.N),

            'Q': (self.H, self.M, self.D),
            'K': (self.H, self.M, self.D),
            'V': (self.H, self.M, self.D),

            'Q1': (self.M, self.N),
            'K1': (self.M, self.N),
            'V1': (self.M, self.N),

            'Q2': (self.M, self.H, self.D),
            'K2': (self.M, self.H, self.D),
            'V2': (self.M, self.H, self.D),

            'K_cache': (self.H, self.P + self.M, self.D),
            'V_cache': (self.H, self.P + self.M, self.D),

            'O': (self.H, self.M, self.D),
            'O1': (self.M, self.H, self.D),
            'O2': (self.M, self.N),

            'C': (self.H, self.M, self.P),
            'C_exp': (self.H, self.M, self.P+self.M),
            'C_div': (self.H, self.M, self.P+self.M),
            'C_sum': (self.H, self.M),
            
            'Q_norm': (self.H, self.M, self.D),
            'K_norm': (self.H, self.M, self.D)
        }

        self.shape_dict = {
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

            'Q_norm': ('H', 'M', 'D'),
            'K_norm': ('H', 'M', 'D')
        }

        self.const_dict = tensor_config.copy()

        # Setup device
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        # if self.device.type != 'cuda':
        #     raise RuntimeError("CUDA device not available. Triton requires CUDA.")
            
        # Create test tensors
        self.create_test_tensors()
        
        # Track temp files for cleanup
        self._temp_files = []
        
    def create_test_tensors(self):
        """Create random test tensors for benchmarking."""
        self.tensors = {}
        for name, shape in self.tensor_shapes.items():
            if name in ['C', 'C_exp', 'C_sum', 'K', 'K1', 'K2', 'O', 'O1', 'Q', 'Q1', 'Q2', 'V', 'V1', 'V2']:  # Output and intermediate tensors
                # Initialize to zero since they're used as accumulators or outputs
                self.tensors[name] = torch.zeros(shape, dtype=torch.float16, device=self.device)
            else:  # Input tensors
                self.tensors[name] = torch.randn(shape, dtype=torch.float16, device=self.device).clamp(-1, 1)*0.01
    
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
                            if 'dummydata' not in ir_expr:
                                expressions.append((ir_id, ir_expr))
                        except:
                            continue
                            
        return expressions
    
    def generate_kernel_code(self, ir_expr: str, constants: Dict[str, int] = None) -> Optional[str]:
        """Generate Triton kernel code from IR expression.
        
        Args:
            ir_expr: IR expression string
            constants: Optional mapping of variable names to constant values
        """
        try:
            kernel_code = convert_ir_to_triton(ir_expr, self.shape_dict, self.const_dict)

            return kernel_code
            
        except Exception as e:
            print(f"Error generating kernel: {e}")
            traceback.print_exc()
            return None
    
    def compile_and_load_kernel(self, kernel_code: str, kernel_id: int) -> Optional[callable]:
        """Compile Triton kernel code and return callable function."""
        try:
            # Create temporary module file
            module_name = f"llama_kernel_{kernel_id}"
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
            
            # Return the module instead of just the function
            # So we can access metadata
            if hasattr(module, 'forward'):
                return module
            else:
                print("Error: Cannot call the kernel")
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
    
    def benchmark_kernel(self, kernel_module, ir_id, warmup_runs: int = 10, benchmark_runs: int = 100) -> float:
        """Benchmark a single kernel and return execution time in milliseconds."""
        try:
            # Get metadata and forward function
            tensor_params = getattr(kernel_module, 'TENSOR_PARAMS', ['K1', 'V1', 'Q1', 
                                                                     'K2', 'V2', 'Q2', 
                                                                     'K', 'V', 'Q',
                                                                     'WK', 'WV', 'WQ',
                                                                     'K_cache', 'V_cache',
                                                                     'C', 'C_sum',
                                                                     'X', 'O', 'O1', 'O2'])
            kernel_fn = kernel_module.forward
            
            # Reset output tensor to zero for each benchmark
            for name in ['O2', 'Q1', 'K1', 'V1', 'Q2', 'K2', 'V2', 'Q', 'K', 'V',
               'C', 'C_exp', 'C_sum', 'C_div', 'O', 'O1', 'X2', 'X_norm']:
                if name in self.tensors:
                    self.tensors[name].zero_()
            
            # Build argument list based on metadata
            args = []
            for param in tensor_params:
                if param in self.tensors:
                    args.append(self.tensors[param])
                else:
                    # Create zero tensor if not exists (for intermediate tensors)
                    if param in self.tensor_shapes:
                        args.append(torch.zeros(self.tensor_shapes[param], dtype=torch.float16, device=self.device))
                    else:
                        raise ValueError(f"Unknown tensor parameter: {param}")
            
            stream = torch.cuda.Stream(self.device)
            # First call to trigger autotune (not counted in warmup)
            kernel_fn(*args)
            torch.cuda.synchronize()
            
            # To see autotune results, run with:
            # TRITON_PRINT_AUTOTUNING=1 python ffn_benchmark.py
            
            # Triton Warmup - now using the best configuration from autotune
            with torch.cuda.stream(stream):
                for _ in range(warmup_runs):
                    kernel_fn(*args)
            stream.synchronize()

            # CUDA Graph Warmup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    kernel_fn(*args)
            # Synchronize before timing
            stream.synchronize()
            
            # Benchmark runs
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with torch.cuda.stream(stream):
                start_event.record()
                for _ in range(benchmark_runs):
                    graph.replay()
                end_event.record()
            stream.synchronize()
            
            # Check for NaN values in O2 tensor
            if 'O2' in self.tensors:
                has_nan = torch.isnan(self.tensors['O2']).any().item()
                if has_nan:
                    print(f"WARNING: NaN values detected in [{ir_id}] O2 tensor!")
            
            # Return average time in milliseconds
            avg_time = (start_event.elapsed_time(end_event)) / benchmark_runs
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
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, "Failed to generate kernel")
            
            # Compile kernel
            kernel_module = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_module is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, "Failed to compile kernel")
            
            # Benchmark kernel
            exec_time = self.benchmark_kernel(kernel_module=kernel_module, ir_id=ir_id)
            
            # Clean up GPU memory after each benchmark
            self.cleanup_gpu()
            
            # Also clean up the loaded module to prevent memory leaks
            module_name = f"llama_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return BenchmarkResult(ir_id, ir_expr, exec_time, self.tensor_config)
            
        except Exception as e:
            # Clean up GPU memory even on error
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, str(e))
    
    def run_all_benchmarks(self, ir_file: str, min_expressions: Optional[int], num: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmarks for all IR expressions in the file."""
        # Parse IR expressions
        expressions = self.parse_ir_file(ir_file)
        
        # Filter by ir_id if min_expressions is provided
        if min_expressions is not None:
            # Find expressions with ir_id >= min_expressions
            filtered_expressions = [(ir_id, expr) for ir_id, expr in expressions if ir_id >= min_expressions]
            
            # If num is specified, take only the first 'num' expressions
            if num:
                filtered_expressions = filtered_expressions[:num]
            
            expressions = filtered_expressions
        
        print(f"Found {len(expressions)} IR expressions to benchmark")
        
        results = []
        # tqdm progress bar with update every 10 items
        with tqdm(total=len(expressions), desc="Benchmarking", unit="IR") as pbar:
            for i, (ir_id, ir_expr) in enumerate(expressions):
                result = self.run_single_benchmark(ir_id, ir_expr)
                results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Update postfix with current status every 10 items
                if (i + 1) % 10 == 0:
                    valid_so_far = sum(1 for r in results if r.error is None)
                    pbar.set_postfix(valid=valid_so_far, errors=len(results)-valid_so_far)
                
                # Clear Triton cache every 100 cases to prevent disk space issues
                # if (i + 1) % 100 == 0:
                #     self.clear_triton_cache()
                #     print(f"\n  Cleared Triton cache after {i + 1} cases")
        
        # Print immediate feedback for errors
            # if result.error:
            #     print(f"  IR {ir_id}: Error - {result.error[:100]}...")

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
                'tensor_config': r.tensor_config,
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
    
    def clear_triton_cache(self):
        """Clear Triton's cache directory to free up disk space."""
        try:
            # Get Triton cache directory
            triton_cache_dir = os.path.expanduser("~/.triton/cache")
            
            if os.path.exists(triton_cache_dir):
                # Remove all files and subdirectories in the cache
                for item in os.listdir(triton_cache_dir):
                    item_path = os.path.join(triton_cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                print(f"  Successfully cleared Triton cache at {triton_cache_dir}")
            else:
                print(f"  Triton cache directory not found at {triton_cache_dir}")
                
        except Exception as e:
            print(f"  Warning: Failed to clear Triton cache: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self._temp_files = []


def run_comprehensive_benchmark(tensor_configs, ir_file, start_expressions, num_expressions, top_k, output_file):
    """Run benchmarks for all tensor shape configurations."""
    all_results = []
    benchmark_instances = []

    print(f"Running comprehensive benchmark with:")
    print(f"  - {len(tensor_configs)} tensor configurations")
    print()
    
    # Initialize output file with empty list
    with open(output_file, 'w') as f:
        json.dump([], f)
    
    for tensor_idx, tensor_config in enumerate(tensor_configs):
        print(f"\nTensor Configuration {tensor_idx + 1}/{len(tensor_configs)}: M={tensor_config['M']}, N={tensor_config['N']}")
        
        # Initialize benchmark with this tensor configuration
        benchmark = RMSBenchmark(tensor_config)
        benchmark_instances.append(benchmark)
        
        try:
            # Run benchmarks for this tensor configuration
            results = benchmark.run_all_benchmarks(ir_file, min_expressions=start_expressions, num=num_expressions)
            
            # Store results with configuration info
            config_results = {
                'tensor_config': tensor_config,
                'results': results
            }
            all_results.append(config_results)
            
            # Save results incrementally
            save_incremental_results(config_results, output_file)
            print(f"  Saved results for configuration {tensor_idx + 1}/{len(tensor_configs)}")
            
        except Exception as e:
            print(f"  Error in configuration: {str(e)}")
            # Save error information
            error_result = {
                'tensor_config': tensor_config,
                'error': str(e),
                'results': []
            }
            all_results.append(error_result)
            save_incremental_results(error_result, output_file)
        
        # Clean up after each tensor configuration
        benchmark.cleanup()
    
    return all_results, benchmark_instances


def save_incremental_results(config_results, output_file):
    """Append results from one configuration to the output file."""
    # Read existing data
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    # Add new results
    if 'error' in config_results and config_results['error']:
        # Handle error case
        existing_data.append({
            'tensor_config': config_results['tensor_config'],
            'error': config_results['error'],
            'results': []
        })
    else:
        # Add all results from this configuration
        for result in config_results['results']:
            existing_data.append({
                'ir_id': result.ir_id,
                'ir_expression': result.ir_expression,
                'execution_time_ms': result.execution_time,
                'tensor_config': result.tensor_config,
                'error': result.error
            })
    
    # Write back to file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def save_comprehensive_results(all_results, output_file):
    """Save all benchmark results with configuration details."""
    data = []
    
    for config_result in all_results:
        if 'error' in config_result and config_result['error']:
            # Handle error case
            data.append({
                'tensor_config': config_result['tensor_config'],
                'error': config_result['error'],
                'results': []
            })
        else:
            for result in config_result['results']:
                data.append({
                    'ir_id': result.ir_id,
                    'ir_expression': result.ir_expression,
                    'execution_time_ms': result.execution_time,
                    'tensor_config': result.tensor_config,
                    'error': result.error
                })
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def print_comprehensive_report(all_results, top_k):
    """Print comprehensive report showing best kernels for each configuration."""
    print("\n" + "="*100)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("="*100)
    
    # Group results by configuration
    for config_idx, config_result in enumerate(all_results):
        tensor_config = config_result['tensor_config']
        results = config_result['results']
        
        print(f"\nConfiguration {config_idx + 1}:")
        print(f"  Tensor Shape: M={tensor_config['M']}, N={tensor_config['N']}")
        
        # Find best kernels for this configuration
        valid_results = [r for r in results if r.error is None and r.execution_time != float('inf')]
        valid_results.sort(key=lambda x: x.execution_time)
        best_kernels = valid_results[:top_k]
        
        if best_kernels:
            print(f"  Top {min(len(best_kernels), top_k)} kernels:")
            for i, result in enumerate(best_kernels):
                print(f"    {i+1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
                if i == 0:  # Show expression for best kernel only
                    print(f"       Expression: {result.ir_expression[:80]}...")
        else:
            print("  No valid kernels found for this configuration")
    
    # Overall best across all configurations
    print("\n" + "="*100)
    print("OVERALL BEST KERNELS ACROSS ALL CONFIGURATIONS")
    print("="*100)
    
    # Flatten all results
    all_valid_results = []
    for config_result in all_results:
        valid_results = [r for r in config_result['results'] if r.error is None and r.execution_time != float('inf')]
        all_valid_results.extend(valid_results)
    
    all_valid_results.sort(key=lambda x: x.execution_time)
    overall_best = all_valid_results[:top_k]
    
    for i, result in enumerate(overall_best):
        print(f"\n{i+1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
        print(f"   Tensor Config: M={result.tensor_config['M']}, N={result.tensor_config['N']}")
        print(f"   Expression: {result.ir_expression[:100]}...")

# def print_ref(tensor_config, benchmark_instances):
#     """Print reference kernel benchmarks using the same tensors from benchmark instances.
    
#     Args:
#         tensor_config: List of tensor configurations
#         benchmark_instances: List of AttaccBenchmark instances corresponding to each config
#     """
#     for idx, (config, benchmark) in enumerate(zip(tensor_config, benchmark_instances)):
#         M = config['M']
#         N = config['N']
#         D = config['D']
#         P = config['P']
#         H = config['H']

#         device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#         torch.cuda.set_device(device)
        
#         ITER = 100

#         # Reuse tensors from benchmark instance
#         X = benchmark.tensors['X']
#         # Note: K_cache and V_cache have different shapes in benchmark vs reference
#         # Benchmark uses (H, P+M, D) while reference expects (1, P+M, H, D)
#         # We need to reshape accordingly
#         cache_K_benchmark = benchmark.tensors['K_cache']  # Shape: (H, P+M, D)
#         cache_V_benchmark = benchmark.tensors['V_cache']  # Shape: (H, P+M, D)

#         cache_K = cache_K_benchmark.permute(1, 0, 2).unsqueeze(0)  # (H, P+M, D) -> (P+M, H, D) -> (1, P+M, H, D)
#         cache_V = cache_V_benchmark.permute(1, 0, 2).unsqueeze(0)  # (H, P+M, D) -> (P+M, H, D) -> (1, P+M, H, D)

#         # Get WQ, WK, WV from benchmark instance
#         WQ = benchmark.tensors['WQ']
#         WK = benchmark.tensors['WK']
#         WV = benchmark.tensors['WV']
        
#         print("\nTesting TensorRT...")
#         block = TensorRT_kernel_RMSNorm(M, N, D, H, cache_K, cache_V, P, WQ, WK, WV).to(device=device)
#         block.half()
#         with torch.no_grad():
#             for _ in range(10):
#                 out = block(X)
#             torch.cuda.synchronize()

#             start_rt = torch.cuda.Event(enable_timing=True)
#             end_rt = torch.cuda.Event(enable_timing=True)

#             start_rt.record()
#             for _ in range(ITER):
#                 out = block(X)
#             end_rt.record()
#             torch.cuda.synchronize()
#             rt_time = start_rt.elapsed_time(end_rt)

#         # Reshape for Flash Decoding which expects (1, P+M, H, D)
#         cache_K = cache_K_benchmark.permute(1, 0, 2).unsqueeze(0)  # (H, P+M, D) -> (P+M, H, D) -> (1, P+M, H, D)
#         cache_V = cache_V_benchmark.permute(1, 0, 2).unsqueeze(0)  # (H, P+M, D) -> (P+M, H, D) -> (1, P+M, H, D)

#         # print("\nTesting Flash Decoding...")
#         # block = Torch_kernel1(N, D, H, cache_K, cache_V, P).to(device=device)
#         # block.half()
#         # with torch.no_grad():
#         #     for _ in range(10):
#         #         out = block(X)
#         #     torch.cuda.synchronize()

#         #     start_flash = torch.cuda.Event(enable_timing=True)
#         #     end_flash = torch.cuda.Event(enable_timing=True)

#         #     start_flash.record()
#         #     for _ in range(ITER):
#         #         out = block(X)
#         #     end_flash.record()
#         #     torch.cuda.synchronize()
#         #     flash_time = start_flash.elapsed_time(end_flash)

#         print("\nTesting Simple Attention...")
#         # For SimpleAttention, use the benchmark tensors directly (already in correct shape)
#         cache_K_simple = benchmark.tensors['K_cache']  # Shape: (H, P+M, D)
#         cache_V_simple = benchmark.tensors['V_cache']  # Shape: (H, P+M, D)
#         simple_block = SimpleAttention(M, N, D, P, cache_K_simple, cache_V_simple, WQ, WK, WV).to(device=device)
#         with torch.no_grad():
#             for _ in range(10):
#                 out = simple_block(X)
#             torch.cuda.synchronize()
        
#             start_simple =torch.cuda.Event(enable_timing=True)
#             end_simple = torch.cuda.Event(enable_timing=True)

#             start_simple.record()
#             for _ in range(ITER):
#                 out = simple_block(X)
#             end_simple.record()
#             torch.cuda.synchronize()
#             simple_time = start_simple.elapsed_time(end_simple)
        
#         print("\n" + "="*100)
#         print("REFERENCE KERNELS TIME")
#         print(f"GPU: {torch.cuda.get_device_name(device)}")
#         print(f"TensorRT execution time: {rt_time / ITER:.3f} ms / iter")
#         # print(f"PyTorch Flash Decoding execution time: {flash_time / ITER:.3f} ms / iter")
#         print(f"Simple manual attention execution time: {simple_time / ITER:.3f} ms / iter")
#         print("="*100)

def main():
    """Main function to run Attacc IR benchmarks."""
    # Configuration
    IR_FILE = "./evaluation/qknorm/qknorm_falcon_cost6_kern2.txt"
    OUTPUT_FILE = "./evaluation/qknorm/qknorm_falcon.json"
    TENSOR_FILE = "./evaluation/falcon_configs.json"
    START_EXPRESSIONS = 0
    NUM_EXPRESSIONS = 10  # Reduced for testing multiple configurations
    TOP_K = 5  # Number of best kernels to report
    
    # Define tensor shape candidates
    # TENSOR_CONFIGS = [
    #     {'M': 16, 'N': 1024, 'P': 512, 'R': 64},      # Original config
    #     {'M': 32, 'N': 1024, 'P': 512, 'R': 64},      # Larger batch
    #     {'M': 16, 'N': 2048, 'P': 1024, 'R': 128},    # Larger model
    #     {'M': 8, 'N': 512, 'P': 256, 'R': 32},        # Smaller model
    # ]
    with open(TENSOR_FILE, 'r') as f:
        TENSOR_CONFIGS = json.load(f)

    parser = argparse.ArgumentParser(description="Run comprehensive Attacc IR benchmarks")
    parser.add_argument('--ir', type=str, default=IR_FILE, help="Path to the IR expressions file")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help="Path to save benchmark results")
    parser.add_argument('--start', type=int, default=START_EXPRESSIONS, help="Start from test case ID (e.g., 1881)")
    parser.add_argument('--num', type=int, default=NUM_EXPRESSIONS, help="Number of expressions to benchmark")
    parser.add_argument('--end', action='store_true', help="Run from start ID to the last test case")
    parser.add_argument('--topk', type=int, default=TOP_K, help="Number of top kernels to report")
    parser.add_argument('--all', action='store_true', help="Run all configurations comprehensively")
    
    # Add options to customize tensor and block configs
    # parser.add_argument('--tensor-configs', type=str, help="JSON file with tensor configurations")
    # parser.add_argument('--block-configs', type=str, help="JSON file with block configurations")

    args = parser.parse_args()
    
    # Validate conflicting arguments
    if args.end and args.num != NUM_EXPRESSIONS:
        print("Error: Cannot use --end and --num together. Use either --num or --end, not both.")
        return
    
    total_expressions = 0
    if args.all:
        with open(args.ir, 'r') as f:
            total_expressions = len(f.readlines())
    elif args.end:
        # When using --end, we don't limit the number, just filter by start ID
        # The actual filtering happens in run_all_benchmarks based on ir_id
        total_expressions = None  # None means no limit
    else:
        total_expressions = args.num

    # Load custom configurations if provided
    # if args.tensor_configs:
        # with open(args.tensor_configs, 'r') as f:
        #     TENSOR_CONFIGS = json.load(f)
    
    # if args.block_configs:
        # with open(args.block_configs, 'r') as f:
        #     BLOCK_CONFIGS = json.load(f)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA device not available. Triton requires CUDA.")
        return
    
    # Run comprehensive benchmarks
    print("Starting comprehensive Attacc benchmarks...")
    all_results, benchmark_instances = run_comprehensive_benchmark(
        TENSOR_CONFIGS, 
        args.ir, 
        args.start, 
        total_expressions, 
        args.topk,
        args.output  # Pass output file for incremental saving
    )
    
    # Results are already saved incrementally, but save again for completeness
    # save_comprehensive_results(all_results, args.output)
    print(f"\nAll results saved to: {args.output}")
    
    # Print comprehensive report
    print_comprehensive_report(all_results, args.topk)
    # print_ref(TENSOR_CONFIGS, benchmark_instances)

if __name__ == "__main__":
    main()