from typing import Dict, Tuple
from IrParser import IRParser
from TritonGen import TritonCodeGen

# Example usage and test
def convert_ir_to_triton(ir_code: str, tensor_shapes: Dict[str, Tuple[int, ...]] = None) -> str:
    """Convert IR code to Triton kernel"""
    parser = IRParser()
    ast = parser.parse(ir_code)
    
    codegen = TritonCodeGen()
    triton_code = codegen.generate(ast, tensor_shapes)
    
    return triton_code

# Test with example IR
if __name__ == "__main__":
    # Example 1: Single parallel loop - 1D grid
    example_1d_ploop = """
    (ploop 0 1024 128 i
        (store (input C) 
               (+ (load (input A) (index (tile i))) 
                  (load (input B) (index (tile i))))
               (index (tile i))))
    """
    
    # Example 2: Nested parallel loops - 2D grid  
    example_2d_ploop = """
    (ploop 0 64 8 i
        (ploop 0 128 16 j
            (store (input C) 
                   (+ (load (input A) (index (tile i) (tile j))) 
                      (load (input B) (index (tile i) (tile j))))
                   (index (tile i) (tile j)))))
    """
    
    # Example 3: Mixed parallel/sequential
    example_mixed = """
    (ploop 0 64 8 i
        (sloop 0 128 16 j
            (store (input C) 
                   (+ (load (input C) (index (tile i))) 
                      (rsum (load (input A) (index (tile i) (tile j))) 1))
                   (index (tile i)))))
    """
    
    print("Example 1 - 1D Parallel Loop (Grid: 8x1):")
    print("=" * 50)
    result1 = convert_ir_to_triton(example_1d_ploop, {'A': (1024,), 'B': (1024,), 'C': (1024,)})
    print(result1)
    
    print("\n\nExample 2 - 2D Parallel Loop (Grid: 8x8):")
    print("=" * 50)
    result2 = convert_ir_to_triton(example_2d_ploop, {'A': (64, 128), 'B': (64, 128), 'C': (64, 128)})
    print(result2)
    
    print("\n\nExample 3 - Mixed Parallel/Sequential:")
    print("=" * 50)
    result3 = convert_ir_to_triton(example_mixed, {'A': (64, 128), 'C': (64,)})
    print(result3)