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
from tqdm import tqdm
import json
import argparse

from convert_module import convert_ir_to_triton

"""
[tensor]
C1
C1_exp
C2

[input]
X
W1
W2

[output]
O
"""
@dataclass
class BenchmarkResult:
    ir_id: int
    ir_expression: str
    execution_time: float
    tensor_config: Dict[str, int]
    block_config: Dict[str, int]
    error: Optional[str] = None
    
    
class GMLPBenchmark:
    def __init__(self, tensor_config: Dict[str, int]):
        """Initialize benchmark with given tensor configuration."""
        # Extract dimensions from config
        self.M = tensor_config['M']
        self.N = tensor_config['N']
        self.K = tensor_config['K']
        
        # Store the config
        self.tensor_config = tensor_config
        
        self.tensor_shapes = {
            # Primary tensors
            'X': (self.M, self.K),
            'W1': (self.K, self.N),
            'W2': (self.K, self.N),
            'O': (self.M, self.N),
            
            # Intermediate tensors (for seq operations)
            'C1': (self.M, self.N),
            'C2': (self.M, self.N),
            'C1_exp': (self.M, self.N),
        }
        
        self.shape_dict = {
            'X': ('M', 'K'),
            'W1': ('K', 'N'),
            'W2': ('K', 'N'),
            'O': ('M', 'N'),
            'C1': ('M', 'N'),
            'C2': ('M', 'N'),
            'C1_exp': ('M', 'N'),
        }

        self.const_dict = tensor_config.copy()

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
            if name in ['C1', 'C1_exp', 'C2']:  # Output and intermediate tensors
                # Initialize to zero since they're used as accumulators or outputs
                self.tensors[name] = torch.zeros(shape, dtype=torch.float16, device=self.device)
            else:  # Input tensors
                self.tensors[name] = torch.randn(shape, dtype=torch.float16, device=self.device)
    
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
            module_name = f"gmlp_kernel_{kernel_id}"
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
    
    def benchmark_kernel(self, kernel_module, block_sizes: Dict[str, int], warmup_runs: int = 10, benchmark_runs: int = 100) -> float:
        """Benchmark a single kernel and return execution time in milliseconds."""
        try:
            # Get metadata and forward function
            tensor_params = getattr(kernel_module, 'TENSOR_PARAMS', ['A', 'B', 'O', 'W', 'X'])
            kernel_fn = kernel_module.forward
            
            # Reset output tensor to zero for each benchmark
            if 'O' in self.tensors:
                self.tensors['O'].zero_()
            
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
            
            # Add block parameters from config
            block_n = block_sizes.get('block_n', 16)
            block_k = block_sizes.get('block_k', 16)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel_fn(*args, block_n=block_n, block_k=block_k)
            
            # Synchronize before timing
            torch.cuda.synchronize()
            
            # Benchmark runs
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(benchmark_runs):
                kernel_fn(*args, block_n=block_n, block_k=block_k)
            end_event.record()
            torch.cuda.synchronize()
            
            # Return average time in milliseconds
            avg_time = (start_event.elapsed_time(end_event)) / benchmark_runs
            return avg_time
            
        except Exception as e:
            print(f"Error benchmarking kernel: {e}")
            traceback.print_exc()
            # Re-raise to let the caller handle it
            raise
    
    def run_single_benchmark(self, ir_id: int, ir_expr: str, block_config: Dict[str, int]) -> BenchmarkResult:
        """Run benchmark for a single IR expression."""
        try:
            # Generate kernel code
            kernel_code = self.generate_kernel_code(ir_expr)
            if kernel_code is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, block_config, "Failed to generate kernel")
            
            # Compile kernel
            kernel_module = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_module is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, block_config, "Failed to compile kernel")
            
            # Benchmark kernel
            exec_time = self.benchmark_kernel(kernel_module, block_config)
            
            # Clean up GPU memory after each benchmark
            self.cleanup_gpu()
            
            # Also clean up the loaded module to prevent memory leaks
            module_name = f"gmlp_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return BenchmarkResult(ir_id, ir_expr, exec_time, self.tensor_config, block_config)
            
        except Exception as e:
            # Clean up GPU memory even on error
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, block_config, str(e))
    
    def run_all_benchmarks(self, ir_file: str, block_config: Dict[str, int], min_expressions: Optional[int], num: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmarks for all IR expressions in the file."""
        # Parse IR expressions
        expressions = self.parse_ir_file(ir_file)
        if num:
            expressions = expressions[min_expressions:min_expressions+num]
        
        print(f"Found {len(expressions)} IR expressions to benchmark")
        
        results = []
        # tqdm progress bar with update every 10 items
        with tqdm(total=len(expressions), desc="Benchmarking", unit="IR") as pbar:
            for i, (ir_id, ir_expr) in enumerate(expressions):
                result = self.run_single_benchmark(ir_id, ir_expr, block_config)
                results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Update postfix with current status every 10 items
                if (i + 1) % 10 == 0:
                    valid_so_far = sum(1 for r in results if r.error is None)
                    pbar.set_postfix(valid=valid_so_far, errors=len(results)-valid_so_far)
        
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
                'block_config': r.block_config,
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


def run_comprehensive_benchmark(tensor_configs, block_configs, ir_file, start_expressions, num_expressions, top_k, output_file):
    """Run benchmarks for all combinations of tensor shapes and block sizes."""
    all_results = []
    
    print(f"Running comprehensive benchmark with:")
    print(f"  - {len(tensor_configs)} tensor configurations")
    print(f"  - {len(block_configs)} block configurations")
    print(f"  - Total combinations: {len(tensor_configs) * len(block_configs)}")
    print()
    
    # Initialize output file with empty list
    with open(output_file, 'w') as f:
        json.dump([], f)
    
    for tensor_idx, tensor_config in enumerate(tensor_configs):
        print(f"\nTensor Configuration {tensor_idx + 1}/{len(tensor_configs)}: M={tensor_config['M']}, N={tensor_config['N']}, K={tensor_config['K']}")
        
        # Initialize benchmark with this tensor configuration
        benchmark = GMLPBenchmark(tensor_config)
        
        for block_idx, block_config in enumerate(block_configs):
            print(f"  Block Configuration {block_idx + 1}/{len(block_configs)}: block_n={block_config['block_n']}, block_k={block_config['block_k']}")
            
            try:
                # Run benchmarks for this combination
                results = benchmark.run_all_benchmarks(ir_file, block_config, min_expressions=start_expressions, num=num_expressions)
                
                # Store results with configuration info
                config_results = {
                    'tensor_config': tensor_config,
                    'block_config': block_config,
                    'results': results
                }
                all_results.append(config_results)
                
                # Save results incrementally after each combination
                save_incremental_results(config_results, output_file)
                print(f"  Saved results for combination {tensor_idx * len(block_configs) + block_idx + 1}/{len(tensor_configs) * len(block_configs)}")
                
            except Exception as e:
                print(f"  Error in combination: {str(e)}")
                # Save error information
                error_result = {
                    'tensor_config': tensor_config,
                    'block_config': block_config,
                    'error': str(e),
                    'results': []
                }
                all_results.append(error_result)
                save_incremental_results(error_result, output_file)
        
        # Clean up after each tensor configuration
        benchmark.cleanup()
    
    return all_results


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
            'block_config': config_results['block_config'],
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
                'block_config': result.block_config,
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
                'block_config': config_result['block_config'],
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
                    'block_config': result.block_config,
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
        block_config = config_result['block_config']
        results = config_result['results']
        
        print(f"\nConfiguration {config_idx + 1}:")
        print(f"  Tensor Shape: M={tensor_config['M']}, N={tensor_config['N']}, K={tensor_config['K']}")
        print(f"  Block Size: block_n={block_config['block_n']}, block_k={block_config['block_k']}")
        
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
        print(f"   Tensor Config: M={result.tensor_config['M']}, N={result.tensor_config['N']}, K={result.tensor_config['K']}")
        print(f"   Block Config: block_n={result.block_config['block_n']}, block_k={result.block_config['block_k']}")
        print(f"   Expression: {result.ir_expression[:100]}...")


def main():
    """Main function to run GMLP IR benchmarks."""
    # Configuration
    IR_FILE = "/home/chani227/triton/part4/benchmark_gmlp/gated_mlp_skipft_cost3_kern2_index17.txt"
    OUTPUT_FILE = "/home/chani227/triton/part4/benchmark_gmlp/benchmark_gmlp_comprehensive.json"
    TENSOR_FILE = "/home/chani227/triton/part4/benchmark_gmlp/tensor_configs.json"
    BLOCK_FILE = "/home/chani227/triton/part4/benchmark_gmlp/block_configs.json"
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
    
    # Define block size candidates
    # BLOCK_CONFIGS = [
    #     {'block_n': 16, 'block_k': 16},    # Small blocks
    #     {'block_n': 32, 'block_k': 32},    # Medium blocks
    #     {'block_n': 64, 'block_k': 64},    # Large blocks
    #     {'block_n': 16, 'block_k': 32},    # Mixed blocks
    #     {'block_n': 32, 'block_k': 16},    # Mixed blocks (reversed)
    # ]
    with open(BLOCK_FILE, 'r') as f:
        BLOCK_CONFIGS = json.load(f)

    parser = argparse.ArgumentParser(description="Run comprehensive GMLP IR benchmarks")
    parser.add_argument('--ir', type=str, default=IR_FILE, help="Path to the IR expressions file")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help="Path to save benchmark results")
    parser.add_argument('--start', type=int, default=START_EXPRESSIONS, help="Start index for expressions")
    parser.add_argument('--num', type=int, default=NUM_EXPRESSIONS, help="Number of expressions to benchmark")
    parser.add_argument('--topk', type=int, default=TOP_K, help="Number of top kernels to report")
    parser.add_argument('--all', action='store_true', help="Run all configurations comprehensively")
    
    args = parser.parse_args()
    
    total_expressions = 0
    if args.all:
        with open(args.ir, 'r') as f:
            total_expressions = len(f.readlines())
    else:
        total_expressions = args.num

    # Add options to customize tensor and block configs
    # parser.add_argument('--tensor-configs', type=str, help="JSON file with tensor configurations")
    # parser.add_argument('--block-configs', type=str, help="JSON file with block configurations")

    args = parser.parse_args()
    
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
    print("Starting comprehensive GMLP benchmarks...")
    all_results = run_comprehensive_benchmark(
        TENSOR_CONFIGS, 
        BLOCK_CONFIGS, 
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


if __name__ == "__main__":
    main()