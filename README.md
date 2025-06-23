# Triton
1. Data Structure
    - NodeType: Type of Operation
    - AstNode: Data class for AST
2. Helper Function
    - IrParser: Convert IR to AST
    - TritonGen: Generate triton kernel with the outcome of IrParser
3. Check List
    - [ ] For `tl.dot` in `matmul`, `allow_tf32` should be `False`  
            (To check the result, `allow_tf32` shoudl be `True`)
    - [ ] Add `elem`, `bcast`, `concat`, `dummy`,`permute3`, `squeeze`, `unsqueeze`
    - [ ] Problem with `for` loop for `fulltile fulltile` index  
            (Due to unused loop variable, tile size mapping doesn't work)
    - [ ] Add `kernel` caller (Need different grid for `ploop` and `sloop`)