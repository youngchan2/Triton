# Triton
1. Data Structure
    - NodeType: Type of Operation
    - AstNode: Data class for AST
2. Helper Function
    - IrParser: Convert IR to AST
    - TritonGen: Generate triton kernel with the outcome of IrParser
3. Check List
    - [x] For `tl.dot` in `matmul`, `allow_tf32` should be `True`  
            (To check the result, `allow_tf32` shoudl be `False`)
    - [x] Add `elem`, `bcast`, `concat`, `dummy`,`permute3`, `squeeze`, `unsqueeze`, `const_tile`
    - [x] Problem with `for` loop for `fulltile fulltile` index  
    - [x] Add `kernel` caller (Need different grid for `ploop` and `sloop`)   
            (Using `forward` to call kernel)
    - [x] Handle with cross-kernel tensors with `tl.store`   
            (Add condition for using `tl.store`)
    - [x] Determine which intermediate tensors should be initialized with `tl.zeros`  
    - [x] Need to optimize the number of argument of kernel   
    - [x] Determine how to apply the tensor information with external file   
    - [ ] Evaluation for the various computation