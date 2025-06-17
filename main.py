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
    example_1d_sloop = """
    (sloop 0 1024 128 i
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
                   (- (load (input A) (index (tile i) (tile j))) 
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
    
    # Example 4: First computation of Attention
    example_att1 = """
    (ploop 0 M tile_m m
        (ploop 0 N tile_n n 
            (store (input C) 
                (*
                    (load (input Q) (index (tile m) (fulltile)))
                    (load (input K) (index (fulltile) (tile n)))
                )
                (index (tile m) (tile n))
            )
        )
    )
    """
    
    example_matmul = """
    (ploop 0 M tile_m m
      (ploop 0 N tile_n n
          (store (input C)
              (*
                  (load (input Q) (index (tile m) (fulltile)))
                  (load (input K) (index (fulltile) (tile n)))
              )
              (index (tile m) (tile n))
          )
      )
    )
    """

    example_exp = """
    (ploop 0 M tile_m m
        (ploop 0 N tile_n n 
            (store (input C_exp)
                (exp (load (input C) (index (tile m) (tile n))))
                (index (tile m) (tile n))
            )
        )
    )
    """

    example_att = """
    (seq
        (ploop 0 M tile_m m
            (ploop 0 N tile_n n 
                (store (input C) 
                    (*
                        (load (input Q) (index (tile m) (fulltile)))
                        (load (input K) (index (fulltile) (tile n)))
                    )
                    (index (tile m) (tile n))
                )
            )
        )
        (ploop 0 M tile_m m
            (ploop 0 N tile_n n 
                (store (input C_exp)
                    (exp (load (input C) (index (tile m) (tile n))))
                    (index (tile m) (tile n))
                )
            )
        )
    )
    """
    # matmul = convert_ir_to_triton(example_matmul, {'A': (512, 256), 'B': (256, 128), 'C': (512, 128)})
    # with open("create_matmul.py", "w") as f:
    #     f.write(matmul)
    
    # Convert example_exp to Triton kernel
    # exp_result = convert_ir_to_triton(example_exp, {'C': (128, 128), 'C_exp': (128, 128)})
    # with open("exp_kernel.py", "w") as f:
    #     f.write(exp_result)

    # Read IR from att.txt file
    # with open("att.txt", "r") as f:
    #     att_ir = f.read()
    
    # # Define tensor shapes for attention computation
    # tensor_shapes = {
    #     'Q': (128, 64),      # Query matrix
    #     'K': (64, 128),      # Key matrix  
    #     'V': (128, 64),      # Value matrix
    #     'C': (128, 128),     # Q @ K^T result
    #     'C_exp': (128, 128), # exp(C)
    #     'C_sum': (128,),     # Row-wise sum of exp(C)
    #     'C_div': (128, 128), # Softmax result
    #     'O': (128, 64)       # Output (attention @ V)
    # }
    
    # # Convert IR to Triton kernel
    # att_result = convert_ir_to_triton(att_ir, tensor_shapes)
    # with open("full_attention_kernel.py", "w") as f:
    #     f.write(att_result)

    # Read LoRA IR
    with open("lora.txt", "r") as f:
        lora_ir = f.read()
    
    # Replace tile sizes in LoRA IR to use smaller blocks
    lora_ir = lora_ir.replace("tile_p", "32")
    lora_ir = lora_ir.replace("tile_n", "32")
    
    # Keep P as a variable instead of replacing it
    # This helps the code generator understand which dimension it refers to
    # lora_ir = lora_ir.replace(" P ", " P ")
    
    # Define tensor shapes for LoRA computation
    # LoRA: O = X @ W + X @ A @ B
    # where A is down projection (N x R) and B is up projection (R x P)
    M, N, P, R = 128, 64, 128, 16  # R is LoRA rank
    
    lora_tensor_shapes = {
        'X': (M, N),      # Input tensor
        'W': (N, P),      # Original weight matrix
        'A': (N, R),      # LoRA down projection
        'B': (R, P),      # LoRA up projection  
        'C': (M, P),      # X @ W result
        'D': (M, R),      # X @ A result
        'E': (M, P),      # D @ B result
        'O': (M, P)       # Final output (C + E)
    }
    
    # Convert LoRA IR to Triton kernel
    lora_result = convert_ir_to_triton(lora_ir, lora_tensor_shapes)
    with open("lora_kernel.py", "w") as f:
        f.write(lora_result)

    # result = convert_ir_to_triton(example_att1, {'Q': (128, 64), 'K': (64, 128), 'C': (128, 128)})
    # with open("att1.py", "w") as f:
    #     f.write(result)

    # print("Example 1 - 1D Parallel Loop (Grid: 8x1):")
    # print("=" * 50)
    # result1 = convert_ir_to_triton(example_1d_ploop, {'A': (1024,), 'B': (1024,), 'C': (1024,)})

    # with open("ploop_1d.py", "w") as f:
    #     f.write(result1)
    
    # print("\n\nExample 2 - 2D Parallel Loop (Grid: 8x8):")
    # print("=" * 50)
    # result2 = convert_ir_to_triton(example_2d_ploop, {'A': (64, 128), 'B': (64, 128), 'C': (64, 128)})

    # with open("ploop_2d.py", "w") as f:
    #     f.write(result2)
    
    # print("\n\nExample 3 - Mixed Parallel/Sequential:")
    # print("=" * 50)
    # result3 = convert_ir_to_triton(example_mixed, {'A': (64, 128), 'C': (64,)})

    # with open("mixed_loop.py", "w") as f:
    #     f.write(result3)