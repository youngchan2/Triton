# Trinity: Three-Dimensional Tensor Optimization via Tile-level Equality Saturation

## Data Structure
    - NodeType: Represent the type of operation in the IR
    - AstNode: Data class for representing the AST (Abstract Syntax Tree)
## Helper Function
    - IrParser: Convert IR into AST format
    - TritonGen: Generate Triton kernel based on the parsed AST
## Development Check List
    - [x] For `tl.dot` in `matmul`, `allow_tf32` should be `True`  
        - Enable `allow_tf32` for performance. 
        - Disable `allow_tf32` to verify correctness
    - [x] Add operator support: `elem`, `bcast`, `concat`, `dummy`,`permute3`, `squeeze`, `unsqueeze`, `const_tile`, `sigmoid`, `sqrt`, `rsqrt`
    - [x] Fix indexing issue for nested `for` loops in `fulltile fulltile`  
    - [x] Add kernel caller with different grid configs for `ploop` and `sloop`  
        - Use `forward` to invoke the kernel
    - [x] Distinguish between on-chip and off-chip tensors to optimize memory usage and performance
        - Support cross-kernel tensors with `tl.store`  
        - Add condition to determine when `tl.store` is used
    - [x] Decide which intermediate tensors should be initialized with `tl.zeros`  
    - [x] Optimize the number of argument of kernel   
    - [x] Support applying tensor metadata from external files
    - [x] Add Evaluation for the various computation  
    - [x] Handle multiple tensors in one `input`&`tensor` operators.
## Evaluation
    - Vanilla
    - PreNorm
    - QKNorm
    - KeyFormer
    - RoCo
## Model Configuration
    - Falcon 7B
    - Llama3 8B
## Target Hardware
    - NVIDIA H100
    - NVIDIA A100
    - NVIDIA A40
    - NVIDIA RTX5090
    - NVIDIA RTX4090