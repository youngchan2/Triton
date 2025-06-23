from typing import Dict, Tuple, List
from NodeType import NodeType
from AstNode import ASTNode

class TritonCodeGen:
    """Generate Triton code from IR AST"""
    
    def __init__(self):
        self.tensor_shapes = {}  # tensor_name -> shape
        self.loop_vars = {}     # loop_var -> (start, end, tile_size, is_parallel)
        self.temp_counter = 0
        self.indent_level = 0
        self.parallel_dims = []  # Track parallel loop dimensions for grid calculation
        self.all_loops = []  # Track all loops (parallel and sequential) for BLOCK parameters
        self.program_id_counter = 0
        self.tensors_used = set()
        self.offset_counter = 0  # Counter for offset variables
        self.mask_counter = 0    # Counter for mask variables
        self.loop_var_to_tensor_dim = {}  # loop_var -> (tensor_name, dimension)
        self.kernel_counter = 0  # Counter for kernel names
        self.generated_kernels = []  # List of (kernel_name, kernel_code, tensors_used, parallel_dims)
        
    def generate(self, ast: ASTNode, tensor_shapes: Dict[str, Tuple[int, ...]] = None) -> str:
        """Generate Triton kernel code from AST"""
        if tensor_shapes:
            self.tensor_shapes = tensor_shapes
            
        # Reset state
        self.kernel_counter = 0
        self.generated_kernels = []
        
        # Check if root is a seq node
        if ast.node_type == NodeType.SEQ:
            # Generate separate kernels for each child
            return self._generate_seq_kernels(ast)
        else:
            # Generate single kernel as before
            return self._generate_single_kernel(ast)
    
    def _has_only_dummy(self, node: ASTNode) -> bool:
        """Check if a node contains only dummy operations"""
        if node.node_type == NodeType.DUMMY:
            return True
        elif node.node_type == NodeType.SEQ:
            # Check if all children are dummy
            return all(self._has_only_dummy(child) for child in node.children)
        elif node.node_type in [NodeType.PLOOP, NodeType.SLOOP]:
            # Check the body (last child)
            if len(node.children) > 4:
                return self._has_only_dummy(node.children[4])
            return False
        else:
            return False
    
    def _flatten_seq(self, node: ASTNode):
        """Recursively flatten nested seq nodes into a list of operations"""
        if node.node_type == NodeType.SEQ:
            operations = []
            for child in node.children:
                operations.extend(self._flatten_seq(child))
            return operations
        elif node.node_type == NodeType.DUMMY:
            # Skip dummy nodes
            return []
        elif self._has_only_dummy(node):
            # Skip nodes that contain only dummy operations
            return []
        else:
            return [node]
    
    def _generate_seq_kernels(self, seq_node: ASTNode) -> str:
        """Generate separate kernels for each operation in a (possibly nested) seq"""
        all_code = ""
        
        # Generate import statements once
        all_code += """import triton
import triton.language as tl
import torch

"""
        
        # Flatten all nested seq nodes
        operations = self._flatten_seq(seq_node)
        
        # Generate a kernel for each operation
        for i, op in enumerate(operations):
            kernel_name = f"kernel_{i}"
            kernel_code = self._generate_single_kernel(op, kernel_name=kernel_name)
            
            # Remove the import statements from individual kernels (they're already at the top)
            kernel_code = kernel_code.replace("import triton\nimport triton.language as tl\nimport torch\n\n", "")
            
            all_code += kernel_code + "\n\n"
            
            # Store kernel info for wrapper generation
            self.generated_kernels.append((
                kernel_name,
                kernel_code,
                self.tensors_used.copy(),
                self.parallel_dims.copy(),
                self.all_loops.copy()
            ))
        
        # Generate wrapper function
        all_code += self._generate_wrapper_function()
        
        return all_code
    
    def _generate_wrapper_function(self) -> str:
        """Generate a wrapper function that calls all kernels sequentially"""
        # Collect all unique tensors used across all kernels
        all_tensors = set()
        # Collect all unique block parameters needed
        all_block_params = set()
        
        for _, _, tensors_used, _, all_loops in self.generated_kernels:
            all_tensors.update(tensors_used)
            # Collect block parameters from loops
            for loop_var, _, _, tile_size in all_loops:
                # Check if tile_size is a variable (not a number)
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    block_param = f"block_{loop_var}"
                    all_block_params.add(block_param)
        
        code = "def forward("
        
        # Generate parameter list
        params = []
        for tensor_name in sorted(all_tensors):
            params.append(f"{tensor_name}")
        
        # Add block size parameters with defaults
        for block_param in sorted(all_block_params):
            params.append(f"{block_param}=16")
        
        code += ", ".join(params) + "):\n"
        code += '    """\n'
        code += '    Wrapper function that executes all kernels sequentially.\n'
        code += '    """\n'
        
        # Call each kernel
        for kernel_name, _, tensors_used, parallel_dims, all_loops in self.generated_kernels:
            # Calculate grid size for this kernel
            if parallel_dims:
                grid_parts = []
                for idx, (loop_var, start, end, tile_size) in enumerate(parallel_dims):
                    # Try to use tensor dimensions for end values
                    end_param = end
                    
                    # Check if it's a variable like M, N, K
                    if end in ['M', 'N', 'K'] or (isinstance(end, str) and end.isalpha()):
                        # Try to infer from tensor shapes
                        if loop_var in self.loop_var_to_tensor_dim:
                            tensor_name, dim = self.loop_var_to_tensor_dim[loop_var]
                            if tensor_name in tensors_used:
                                end_param = f"{tensor_name}.shape[{dim}]"
                        
                        # If still not resolved, use the loop_var_to_tensor_dim mapping
                        if end_param == end and loop_var in self.loop_var_to_tensor_dim:
                            tensor_name, dim = self.loop_var_to_tensor_dim[loop_var]
                            # Find any tensor with the same number of dimensions
                            for tensor in sorted(tensors_used):
                                if tensor in self.tensor_shapes:
                                    shape = self.tensor_shapes[tensor]
                                    if len(shape) > dim:
                                        end_param = f"{tensor}.shape[{dim}]"
                                        break
                        
                        # If still not resolved, fall back to heuristic
                        if end_param == end:  # Still unchanged
                            # Look for a 2D tensor to get dimensions from
                            found = False
                            for tensor in sorted(tensors_used):
                                if tensor in self.tensor_shapes:
                                    shape = self.tensor_shapes[tensor]
                                    if len(shape) >= 2:
                                        # Try to be smarter: if we have loop_var_to_tensor_dim info, use it
                                        if loop_var in self.loop_var_to_tensor_dim:
                                            _, preferred_dim = self.loop_var_to_tensor_dim[loop_var]
                                            if preferred_dim < len(shape):
                                                end_param = f"{tensor}.shape[{preferred_dim}]"
                                                found = True
                                                break
                                        # Otherwise fall back to index-based heuristic
                                        if not found:
                                            if idx == 0:  # First parallel dimension
                                                end_param = f"{tensor}.shape[0]"
                                            elif idx == 1:  # Second parallel dimension
                                                end_param = f"{tensor}.shape[1]"
                                            found = True
                                            break
                            
                            # If still not found, try to use any tensor's dimension
                            if not found:
                                for tensor in sorted(tensors_used):
                                    if tensor in self.tensor_shapes:
                                        shape = self.tensor_shapes[tensor]
                                        if len(shape) > idx:
                                            end_param = f"{tensor}.shape[{idx}]"
                                            break
                    
                    # Use the block parameter for tile size
                    if isinstance(tile_size, str) and not tile_size.isdigit():
                        # Use block parameter variable
                        block_var = f"block_{loop_var}"
                    else:
                        # Use numeric value
                        block_var = tile_size
                    grid_parts.append(f"({end_param} - {start} + {block_var} - 1) // {block_var}")
                
                grid_calc = "(" + ", ".join(grid_parts) + ",)"
                
                code += f"    # Launch {kernel_name}\n"
                code += f"    grid = {grid_calc}\n"
            else:
                code += f"    # Launch {kernel_name}\n"
                code += f"    grid = (1,)\n"
            
            # Generate kernel call
            code += f"    {kernel_name}[grid](\n"
            
            # Add parameters for this kernel
            kernel_params = []
            for tensor in sorted(tensors_used):
                # Pointer
                kernel_params.append(f"        {tensor}")
                
                # Shape parameters
                if tensor in self.tensor_shapes:
                    num_dims = len(self.tensor_shapes[tensor])
                else:
                    num_dims = max(len(parallel_dims), 2)  # Default to 2D
                
                for dim in range(num_dims):
                    kernel_params.append(f"        {tensor}.shape[{dim}]")
                
                # Stride parameters
                for dim in range(num_dims):
                    kernel_params.append(f"        {tensor}.stride({dim})")
            
            # Add block size parameters for all loops
            for loop_var, _, _, tile_size in all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    # Use block parameter variable
                    kernel_params.append(f"        block_{loop_var}")
                else:
                    # Use numeric value
                    kernel_params.append(f"        {tile_size}")
            
            code += ",\n".join(kernel_params)
            code += "\n    )\n\n"
        
        code += "    # Return output tensors if needed\n"
        code += "    # This depends on your specific use case\n"
        code += "    pass\n"
        
        return code
    
    def _generate_single_kernel(self, ast: ASTNode, kernel_name: str = "kernel") -> str:
        """Generate a single Triton kernel from AST"""
        # Reset state for this kernel
        self.parallel_dims = []
        self.all_loops = []
        self.program_id_counter = 0
        self.tensors_used = set()
        self.temp_counter = 0
        self.offset_counter = 0
        self.mask_counter = 0
        self.loop_var_to_tensor_dim = {}
        self.loop_vars = {}
        
        # First pass: collect all tensors used in the AST
        self._collect_tensors(ast)
        
        # Second pass: collect all loops to determine BLOCK parameters
        self._collect_all_loops(ast)
        
        # Third pass: analyze loop contexts to map loop vars to tensor dimensions
        self._analyze_loop_contexts(ast)

        # Generate kernel function
        kernel_code = self._generate_kernel_header(kernel_name)

        self.indent_level = 1  # Set initial indent level for kernel body
        kernel_code += self._generate_node(ast)
        self.indent_level = 0  # Reset indent level
        
        return kernel_code
    
    def _collect_tensors(self, node: ASTNode):
        """Collect all tensor names used in the AST"""
        if node.node_type == NodeType.INPUT or node.node_type == NodeType.OUTPUT:
            tensor_name = node.children[0].value
            self.tensors_used.add(tensor_name)

        for child in node.children:
            if isinstance(child, ASTNode):
                self._collect_tensors(child)

    def _calculate_grid_size(self) -> str:
        """Calculate grid size based on parallel dimensions"""
        if not self.parallel_dims:
            return "grid_size = (1,)"
        
        grid_calc = "grid_size = ("
        for i, (loop_var, start, end, tile_size) in enumerate(self.parallel_dims):
            if i > 0:
                grid_calc += ", "
            # Grid size = total_iterations / tile_size
            grid_calc += f"({end} - {start}) // {tile_size}"
        grid_calc += ",)"
        
        return grid_calc
    
    def _generate_kernel_header(self, kernel_name: str = "kernel") -> str:
        # Generate Triton kernel function header with shape and stride parameters
        params = []
        for name in sorted(self.tensors_used):
            # pointer
            params.append(f"{name}_ptr")
            
            # Determine number of dimensions based on tensor_shapes or parallel_dims
            num_dims = 2  # default
            if self.tensor_shapes and name in self.tensor_shapes:
                num_dims = len(self.tensor_shapes[name])
            else:
                # Infer from parallel dimensions
                num_dims = max(len(self.parallel_dims), 1)
            
            # Shape parameters
            for dim in range(num_dims):
                params.append(f"{name}_dim{dim}: tl.constexpr")
            
            # Stride parameters
            for dim in range(num_dims):
                params.append(f"{name}_stride{dim}: tl.constexpr")
        # block sizes for all loops (parallel and sequential)
        for loop_var, start, end, tile_size in self.all_loops:
            params.append(f"BLOCK_{loop_var.upper()}: tl.constexpr")
        params_str = ",\n    ".join(params)
        header = f"""import triton
import triton.language as tl
import torch

@triton.jit
def {kernel_name}(
    {params_str}
):
"""
        return header
    
    def _generate_node(self, node: ASTNode) -> str:
        """Generate code for a single AST node"""
        if node.node_type == NodeType.PLOOP:
            return self._generate_ploop(node)
        elif node.node_type == NodeType.SLOOP:
            return self._generate_sloop(node)
        elif node.node_type == NodeType.SEQ:
            # If we encounter a nested seq, just generate its children inline
            # (This shouldn't happen at the top level since we handle that separately)
            return self._generate_seq(node)
        elif node.node_type == NodeType.LOAD:
            return self._generate_load(node)
        elif node.node_type == NodeType.STORE:
            return self._generate_store(node)
        elif node.node_type == NodeType.ADD:
            return self._generate_binary_op(node, "+")
        elif node.node_type == NodeType.SUB:
            return self._generate_binary_op(node, "-")
        elif node.node_type == NodeType.MUL:
            return self._generate_binary_op(node, "*")
        elif node.node_type == NodeType.DIV:
            return self._generate_binary_op(node, "/")
        elif node.node_type == NodeType.MATMUL:
            return self._generate_matmul(node)
        elif node.node_type == NodeType.EXP:
            return self._generate_unary_op(node, "tl.exp")
        elif node.node_type == NodeType.RSUM:
            return self._generate_reduce_sum(node)
        elif node.node_type == NodeType.BCAST:
            return self._generate_broadcast(node)
        elif node.node_type == NodeType.CONCAT:
            return self._generate_concat(node)
        elif node.node_type == NodeType.INPUT:
            return self._generate_input(node)
        elif node.node_type == NodeType.OUTPUT:
            return self._generate_output(node)
        elif node.node_type == NodeType.NUM:
            return str(node.value)
        elif node.node_type == NodeType.VAR:
            return node.value
        elif node.node_type == NodeType.DUMMY:
            # Dummy node is a no-op placeholder
            indent = '    ' * self.indent_level
            return f"{indent}pass  # dummy node"
        elif node.node_type == NodeType.PERMUTE3:
            return self._generate_permute3(node)
        elif node.node_type == NodeType.SQUEEZE:
            return self._generate_squeeze(node)
        elif node.node_type == NodeType.UNSQUEEZE:
            return self._generate_unsqueeze(node)
        else:
            return f"# TODO: Implement {node.node_type.value}"
    
    def _generate_concat(self, node: ASTNode) -> str:
        """Generate concatenation operation
        
        The concat node has at least 3 children:
        1. First tensor to concatenate
        2. Second tensor to concatenate
        3. Axis along which to concatenate
        
        In the IR expressions:
        - (concat (load X ...) (load X ...) 1) means concatenate along axis 1
        - (concat (* A B) (load W ...) 0) means concatenate along axis 0
        
        Since Triton doesn't have a direct concat operation, we handle this
        by recognizing the pattern where concat is used in matmul operations
        and transforming it into separate computations.
        """
        if len(node.children) < 3:
            raise ValueError("concat requires at least 3 arguments: tensor1, tensor2, axis")
        
        # Get the axis
        axis = int(self._generate_node(node.children[-1]))
        
        # For the LoRA patterns, concat is used in a specific way:
        # 1. Two concat operations are used in a matmul
        # 2. This represents a block matrix multiplication
        
        # Store concat information in the node for parent operations to handle
        node.is_concat = True
        node.concat_axis = axis
        node.concat_children = node.children[:-1]  # All children except the axis
        
        # Return a special marker that _generate_matmul can recognize
        # This is a workaround since concat needs special handling in matmul context
        return "__CONCAT_NODE__"
    
    def _generate_ploop(self, node: ASTNode) -> str:
        """Generate parallel loop code - executed across GPU grid dimensions"""
        # (ploop start end tile_size loop_var body)
        start = self._generate_node(node.children[0])
        end = self._generate_node(node.children[1])
        tile_size = self._generate_node(node.children[2])
        loop_var = node.children[3].value
        body = node.children[4]
        
        # Store loop info for tile generation
        self.loop_vars[loop_var] = (start, end, tile_size, True)  # True = parallel
        
        # For parallel loops, map to grid dimensions
        # Find the grid dimension for this loop variable
        grid_dim = -1
        for idx, (lv, _, _, _) in enumerate(self.parallel_dims):
            if lv == loop_var:
                grid_dim = idx
                break
        
        # If not found (shouldn't happen), use the current count
        if grid_dim == -1:
            grid_dim = len(self.parallel_dims)
            self.parallel_dims.append((loop_var, start, end, tile_size))
        
        indent = '    ' * self.indent_level
        # Use BLOCK_SIZE parameter instead of hardcoded tile_size
        block_param = f"BLOCK_{loop_var.upper()}"
        
        # Determine the end value - use tensor dimension parameter if available
        end_param = self._get_end_param(loop_var, end)
        
        code = f"""{indent}# Parallel loop {loop_var} from {start} to {end_param} with tile size {block_param}
{indent}# Executed across grid dimension {grid_dim}
{indent}{loop_var} = {start} + tl.program_id({grid_dim}) * {block_param}
{indent}
{indent}# Boundary check for parallel execution
{indent}if {loop_var} < {end_param}:
"""
        self.indent_level += 1
        code += self._generate_node(body)
        self.indent_level -= 1
        
        return code
    
    def _generate_sloop(self, node: ASTNode) -> str:
        """Generate sequential loop code - executed as traditional for loop"""
        # (sloop start end tile_size loop_var body)
        start = self._generate_node(node.children[0])
        end = self._generate_node(node.children[1])
        tile_size = self._generate_node(node.children[2])
        loop_var = node.children[3].value
        body = node.children[4]
        
        # Check if body is only dummy - skip loop generation
        if body.node_type == NodeType.DUMMY:
            indent = '    ' * self.indent_level
            return f"{indent}# Skipped empty sloop with dummy body\n"
        
        # Store loop info for tile generation
        self.loop_vars[loop_var] = (start, end, tile_size, False)  # False = sequential
        
        indent = '    ' * self.indent_level
        # Use BLOCK_SIZE parameter instead of hardcoded tile_size
        block_param = f"BLOCK_{loop_var.upper()}"
        
        # Determine the end value - use tensor dimension parameter if available
        end_param = self._get_end_param(loop_var, end)
        
        code = f"""{indent}# Sequential loop {loop_var} from {start} to {end_param} with tile size {block_param}
{indent}for {loop_var} in range({start}, {end_param}, {block_param}):
"""
        self.indent_level += 1
        code += self._generate_node(body)
        self.indent_level -= 1
        
        return code
    
    def _generate_seq(self, node: ASTNode) -> str:
        """Generate sequence of operations (for nested seq nodes only)"""
        code = ""
        for child in node.children:
            # Skip dummy nodes in sequences
            if child.node_type == NodeType.DUMMY:
                continue
            code += self._generate_node(child) + "\n"
        return code
    
    def _generate_load(self, node: ASTNode) -> str:
        """Generate load operation"""
        # (load tensor index)
        tensor_node = node.children[0]
        index_node = node.children[1]
        
        tensor_name = tensor_node.children[0].value
        offset_expr, mask_expr = self._generate_index(index_node, tensor_name)
        
        # Generate unique variable names for offset and mask
        offset_var = f"offset_{self.offset_counter}"
        mask_var = f"mask_{self.mask_counter}"
        self.offset_counter += 1
        self.mask_counter += 1
        
        temp_var = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        # Store the temp variable mapping for later use
        node.temp_var = temp_var
        
        # Generate code with separate offset and mask variables
        # Use proper indentation based on current context
        indent_str = '    ' * self.indent_level
        code = f"{indent_str}{offset_var} = {offset_expr}\n"
        code += f"{indent_str}{mask_var} = {mask_expr}\n"
        code += f"{indent_str}{temp_var} = tl.load({tensor_name}_ptr + {offset_var}, mask={mask_var}, other=0.0)"
        
        return code
    
    def _generate_store(self, node: ASTNode) -> str:
        """Generate store operation"""
        # (store tensor val index)
        tensor_node = node.children[0]
        val_node = node.children[1]
        index_node = node.children[2]
        
        tensor_name = tensor_node.children[0].value
        
        # First generate any load operations in the value expression
        code = ""
        
        # Handle the value expression generation
        if self._contains_loads(val_node) or self._contains_reduce_sum(val_node):
            code += self._generate_loads_separately(val_node)
            
            # Check if we have nested matmuls and handle them
            if self._contains_nested_matmul(val_node):
                # Find the nested matmul: X @ (A @ B)
                # We need to generate A @ B as a temporary first
                code += self._generate_nested_matmul_temps(val_node)
            
            value_expr = self._generate_node_without_loads(val_node)
        else:
            value_expr = self._generate_node(val_node)
        
        offset_expr, mask_expr = self._generate_index(index_node, tensor_name)
        
        # Generate unique variable names for offset and mask
        offset_var = f"offset_{self.offset_counter}"
        mask_var = f"mask_{self.mask_counter}"
        self.offset_counter += 1
        self.mask_counter += 1
        
        # Generate code with separate offset and mask variables
        # Use proper indentation based on current context
        indent_str = '    ' * self.indent_level
        code += f"{indent_str}{offset_var} = {offset_expr}\n"
        code += f"{indent_str}{mask_var} = {mask_expr}\n"
        code += f"{indent_str}tl.store({tensor_name}_ptr + {offset_var}, {value_expr}, mask={mask_var})"
        
        return code
    
    def _generate_index(self, node: ASTNode, tensor_name: str) -> Tuple[str, str]:
        """Generate index calculation for memory access"""
        # (index tile1 tile2 ...)
        offsets = []
        masks = []
        
        for i, child in enumerate(node.children):
            if child.node_type == NodeType.TILE:
                loop_var = child.children[0].value
                if loop_var in self.loop_vars:
                    start, end, tile_size, is_parallel = self.loop_vars[loop_var]
                    
                    # Use BLOCK_SIZE parameter instead of hardcoded tile_size
                    block_param = f"BLOCK_{loop_var.upper()}"
                    # Get end parameter (tensor dimension)
                    end_param = self._get_end_param(loop_var, end)
                    
                    if is_parallel:
                        # For parallel loops, create tile range from program_id * tile_size
                        offset_expr = f"{loop_var} + tl.arange(0, {block_param})"
                        mask_expr = f"({loop_var} + tl.arange(0, {block_param})) < {end_param}"
                    else:
                        # For sequential loops, use regular indexing
                        offset_expr = f"{loop_var} + tl.arange(0, {block_param})"
                        mask_expr = f"({loop_var} + tl.arange(0, {block_param})) < {end_param}"
                    
                    offsets.append(offset_expr)
                    masks.append(mask_expr)
            elif child.node_type == NodeType.FULLTILE:
                # For fulltile, we need to determine the appropriate dimension
                # This represents the full dimension being accessed
                # We'll use the tensor dimension parameter
                fulltile_dim = f"{tensor_name}_dim{i}"
                offsets.append(f"tl.arange(0, {fulltile_dim})")
                masks.append(f"tl.arange(0, {fulltile_dim}) < {fulltile_dim}")
            elif child.node_type == NodeType.ELEM:
                # For elem indexing, use the loop variable directly without creating a range
                # (elem n) means use n as a scalar index
                if child.children:
                    elem_var = child.children[0].value
                    # This is a scalar index, not a range
                    offsets.append(elem_var)
                    masks.append("True")  # No mask needed for scalar index
        
        # Generate multi-dimensional indexing
        if len(offsets) == 1:
            # For 1D tensors, still need to multiply by stride
            offset_str = f"({offsets[0]}) * {tensor_name}_stride0"
            mask_str = masks[0]
        else:
            # Multi-dimensional indexing: i[:,None]*tensor_stride0 + j[None,:]*tensor_stride1
            offset_parts = []
            mask_parts = []
            
            # Check which dimensions are scalar (elem) vs array (tile/fulltile)
            is_scalar = [mask == "True" for mask in masks]
            
            for dim, off in enumerate(offsets):
                stride_term = f" * {tensor_name}_stride{dim}"
                
                if is_scalar[dim]:
                    # Scalar index - no broadcasting needed
                    offset_parts.append(f"({off}){stride_term}")
                    mask_parts.append("True")
                else:
                    # Array index - need broadcasting
                    # Calculate how many None dimensions to add
                    nones = []
                    for i in range(len(offsets)):
                        if i == dim:
                            nones.append(":")
                        elif not is_scalar[i]:
                            nones.append("None")
                    
                    if len([n for n in nones if n == ":"]) == 1 and len([n for n in nones if n == "None"]) == 0:
                        # Only one dimension, no broadcasting needed
                        offset_parts.append(f"({off}){stride_term}")
                        mask_parts.append(f"{masks[dim]}")
                    else:
                        # Need broadcasting
                        broadcast_expr = "[" + ", ".join(nones) + "]"
                        offset_parts.append(f"({off}){broadcast_expr}{stride_term}")
                        mask_parts.append(f"({masks[dim]}){broadcast_expr}")
            
            offset_str = " + ".join(offset_parts)
            # Create combined mask - only multiply non-True masks
            non_true_masks = [m for m in mask_parts if m != "True"]
            if non_true_masks:
                mask_str = " * ".join(non_true_masks)
            else:
                mask_str = "True"
        
        return offset_str, mask_str
    
    def _generate_binary_op(self, node: ASTNode, op: str) -> str:
        """Generate binary operation"""
        # Check if children are load operations that have temp_var
        left_child = node.children[0]
        right_child = node.children[1]
        
        # Use temp_var if available (from load operations), otherwise generate normally
        if hasattr(left_child, 'temp_var'):
            left = left_child.temp_var
        else:
            left = self._generate_node(left_child)
            
        if hasattr(right_child, 'temp_var'):
            right = right_child.temp_var
        else:
            right = self._generate_node(right_child)
            
        return f"({left} {op} {right})"
    
    def _generate_unary_op(self, node: ASTNode, op: str) -> str:
        """Generate unary operation"""
        child = node.children[0]
        
        # Use temp_var if available (from load operations), otherwise generate normally
        if hasattr(child, 'temp_var'):
            operand = child.temp_var
        else:
            operand = self._generate_node(child)
            
        return f"{op}({operand})"
    
    def _generate_matmul(self, node: ASTNode) -> str:
        """Generate matrix multiplication"""
        # Check if children are load operations that have temp_var
        left_child = node.children[0]
        right_child = node.children[1]
        
        # Check if either child is a concat operation
        left_is_concat = (hasattr(left_child, 'is_concat') and left_child.is_concat) or \
                         (left_child.node_type == NodeType.CONCAT)
        right_is_concat = (hasattr(right_child, 'is_concat') and right_child.is_concat) or \
                          (right_child.node_type == NodeType.CONCAT)
        
        # Handle the special case where both operands are concat
        # This represents a block matrix multiplication
        if left_is_concat and right_is_concat:
            # Pattern: (* (concat A B axis1) (concat C D axis2))
            # This can be decomposed based on the axes
            
            # For the LoRA pattern:
            # (* (concat X X 1) (concat (A@B) W 0))
            # = X @ (A@B) + X @ W
            
            # Get axis information
            if hasattr(left_child, 'concat_axis'):
                left_axis = left_child.concat_axis
                left_children = left_child.concat_children
            else:
                # Direct NodeType.CONCAT node
                left_axis = int(self._generate_node(left_child.children[-1]))
                left_children = left_child.children[:-1]
                
            if hasattr(right_child, 'concat_axis'):
                right_axis = right_child.concat_axis
                right_children = right_child.concat_children
            else:
                # Direct NodeType.CONCAT node
                right_axis = int(self._generate_node(right_child.children[-1]))
                right_children = right_child.children[:-1]
            
            if left_axis == 1 and right_axis == 0:
                # This is the pattern we see in the LoRA expressions
                # Generate separate matmuls and add them
                
                # Get the concatenated tensors
                left_tensors = []
                for child in left_children:
                    if hasattr(child, 'temp_var'):
                        left_tensors.append(child.temp_var)
                    else:
                        left_tensors.append(self._generate_node(child))
                
                right_tensors = []
                for child in right_children:
                    if hasattr(child, 'temp_var'):
                        right_tensors.append(child.temp_var)
                    else:
                        right_tensors.append(self._generate_node(child))
                
                # For the pattern (concat X X 1) @ (concat T1 T2 0)
                # Result = X @ T1 + X @ T2
                if len(left_tensors) == 2 and len(right_tensors) == 2:
                    # Assuming left_tensors[0] == left_tensors[1] (both are X)
                    matmul1 = f"tl.dot({left_tensors[0]}, {right_tensors[0]}, allow_tf32=False)"
                    matmul2 = f"tl.dot({left_tensors[1]}, {right_tensors[1]}, allow_tf32=False)"
                    return f"({matmul1} + {matmul2})"
                else:
                    # Fallback for other concat patterns
                    raise NotImplementedError(f"Concat pattern with {len(left_tensors)} x {len(right_tensors)} tensors not supported")
            else:
                raise NotImplementedError(f"Concat with axes {left_axis}, {right_axis} not supported in matmul")
        
        # Normal matmul without concat
        if hasattr(left_child, 'temp_var'):
            left = left_child.temp_var
        else:
            left = self._generate_node(left_child)
            
        if hasattr(right_child, 'temp_var'):
            right = right_child.temp_var
        else:
            right = self._generate_node(right_child)
            
        # For DNN compiler, use tl.dot with allow_tf32=False for exact computation
        return f"tl.dot({left}, {right}, allow_tf32=False)"
    
    def _generate_reduce_sum(self, node: ASTNode) -> str:
        """Generate reduce sum operation"""
        # Check if the child is a load operation
        child = node.children[0]
        axis = self._generate_node(node.children[1])
        
        if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
            # Use the temp variable if it was already generated
            return f"tl.sum({child.temp_var}, axis={axis})"
        else:
            # Generate the child expression
            tensor = self._generate_node(child)
            return f"tl.sum({tensor}, axis={axis})"
    
    def _generate_broadcast(self, node: ASTNode) -> str:
        """Generate broadcast operation
        
        The bcast node has two children:
        1. The tensor to broadcast (usually a load operation)
        2. The dimension along which to broadcast
        
        In Triton, broadcasting a 1D tensor to 2D is done by adding [:, None] or [None, :]
        depending on the broadcast dimension.
        """
        # Get the tensor to broadcast
        tensor_child = node.children[0]
        broadcast_dim = int(self._generate_node(node.children[1]))
        
        # Check if the child is a load operation with a temp_var
        if hasattr(tensor_child, 'temp_var'):
            tensor_expr = tensor_child.temp_var
        else:
            tensor_expr = self._generate_node(tensor_child)
        
        # In Triton, to broadcast:
        # - Along dimension 0 (rows): use [None, :]
        # - Along dimension 1 (columns): use [:, None]
        if broadcast_dim == 0:
            return f"{tensor_expr}[None, :]"
        elif broadcast_dim == 1:
            return f"{tensor_expr}[:, None]"
        else:
            # For higher dimensions, we'd need more complex indexing
            raise NotImplementedError(f"Broadcast along dimension {broadcast_dim} not yet implemented")
    
    def _indent(self, code: str) -> str:
        """Add indentation to code"""
        lines = code.split('\n')
        indented = []
        for line in lines:
            if line.strip():
                indented.append('    ' * (self.indent_level + 1) + line)
            else:
                indented.append(line)
        return '\n'.join(indented)
    
    def _get_end_param(self, loop_var: str, end_value: str) -> str:
        """Get the appropriate parameter name for the end value"""
        # Check if we have a mapping for this loop variable
        if loop_var in self.loop_var_to_tensor_dim:
            tensor_name, dim = self.loop_var_to_tensor_dim[loop_var]
            return f"{tensor_name}_dim{dim}"
        
        # Default to the literal value if no mapping found
        return end_value
    
    def _analyze_loop_contexts(self, node: ASTNode, current_loops=None, depth=0):
        """Analyze loop contexts to determine which tensor dimensions each loop variable corresponds to"""
        if current_loops is None:
            current_loops = []
        
        if node.node_type in [NodeType.PLOOP, NodeType.SLOOP]:
            # Extract loop variable from 4th argument
            loop_var = node.children[3].value
            
            # Add current loop to the context
            new_loops = current_loops + [(loop_var, depth)]
            
            # Analyze body with updated context
            if len(node.children) > 4:
                self._analyze_loop_contexts(node.children[4], new_loops, depth + 1)
        
        elif node.node_type == NodeType.LOAD or node.node_type == NodeType.STORE:
            # Found a memory access - analyze its index pattern
            tensor_node = node.children[0]
            index_node = node.children[1] if node.node_type == NodeType.LOAD else node.children[2]
            
            if tensor_node.node_type == NodeType.INPUT or tensor_node.node_type == NodeType.OUTPUT:
                tensor_name = tensor_node.children[0].value
                
                # Analyze index to see which loop variables access which dimensions
                self._analyze_index_pattern(index_node, tensor_name, current_loops)
        
        # Continue analyzing children
        for child in node.children:
            if isinstance(child, ASTNode):
                self._analyze_loop_contexts(child, current_loops, depth)
    
    def _analyze_index_pattern(self, index_node: ASTNode, tensor_name: str, current_loops):
        """Analyze index pattern to map loop variables to tensor dimensions"""
        if index_node.node_type == NodeType.INDEX:
            # Go through each tile in the index
            for dim, tile in enumerate(index_node.children):
                if tile.node_type == NodeType.TILE and tile.children:
                    # Get the loop variable used in this tile
                    loop_var = tile.children[0].value
                    
                    # Map this loop variable to the tensor dimension
                    if loop_var not in self.loop_var_to_tensor_dim:
                        self.loop_var_to_tensor_dim[loop_var] = (tensor_name, dim)
    
    def _generate_input(self, node: ASTNode) -> str:
        """Generate code for input tensor reference"""
        # (input X) -> just return the tensor name
        if node.children:
            return node.children[0].value
        return "input"
    
    def _generate_output(self, node: ASTNode) -> str:
        """Generate code for output tensor reference"""
        # (output O) -> just return the tensor name
        if node.children:
            return node.children[0].value
        return "output"
    
    def _collect_all_loops(self, node: ASTNode):
        """Collect all loop information for BLOCK parameters"""
        if node.node_type == NodeType.PLOOP:
            # Extract loop info: (ploop start end tile_size loop_var body)
            start = node.children[0].value if node.children[0].node_type == NodeType.NUM else "0"
            end = node.children[1].value if node.children[1].node_type == NodeType.NUM else "N"
            # Get tile size - preserve variable names like tile_p, tile_n
            if node.children[2].node_type == NodeType.NUM:
                tile_size = node.children[2].value
            else:
                # It's a variable name like tile_p, tile_n
                tile_size = node.children[2].value if hasattr(node.children[2], 'value') else str(node.children[2])
            loop_var = node.children[3].value
            
            # Check if this loop variable already exists in parallel_dims
            loop_var_exists = any(lv == loop_var for lv, _, _, _ in self.parallel_dims)
            
            # Only add if not already present
            if not loop_var_exists:
                self.parallel_dims.append((loop_var, str(start), str(end), str(tile_size)))
            
            # Also add to all_loops if not already present
            all_loop_exists = any(lv == loop_var for lv, _, _, _ in self.all_loops)
            if not all_loop_exists:
                self.all_loops.append((loop_var, str(start), str(end), str(tile_size)))
            
            # Continue collecting in body
            if len(node.children) > 4:
                self._collect_all_loops(node.children[4])
        elif node.node_type == NodeType.SLOOP:
            # Check if body is dummy - skip if so
            body = node.children[4] if len(node.children) > 4 else None
            if body and body.node_type == NodeType.DUMMY:
                # Skip loops with dummy body
                return
                
            # Sequential loops are NOT parallel - don't add them to parallel_dims
            # But add to all_loops for BLOCK parameters
            start = node.children[0].value if node.children[0].node_type == NodeType.NUM else "0"
            end = node.children[1].value if node.children[1].node_type == NodeType.NUM else "N"
            # Get tile size - preserve variable names like tile_p, tile_n
            if node.children[2].node_type == NodeType.NUM:
                tile_size = node.children[2].value
            else:
                # It's a variable name like tile_p, tile_n
                tile_size = node.children[2].value if hasattr(node.children[2], 'value') else str(node.children[2])
            loop_var = node.children[3].value
            
            # Add to all_loops if not already present
            all_loop_exists = any(lv == loop_var for lv, _, _, _ in self.all_loops)
            if not all_loop_exists:
                self.all_loops.append((loop_var, str(start), str(end), str(tile_size)))
            
            # Continue collecting in body
            if len(node.children) > 4:
                self._collect_all_loops(node.children[4])
        else:
            # Recursively collect from children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._collect_all_loops(child)
    
    def _contains_nested_matmul(self, node: ASTNode) -> bool:
        """Check if node contains nested matmul operations"""
        if node.node_type == NodeType.MATMUL:
            # Check if any child is also a matmul
            for child in node.children:
                if isinstance(child, ASTNode) and child.node_type == NodeType.MATMUL:
                    return True
        
        # Check children recursively
        for child in node.children:
            if isinstance(child, ASTNode) and self._contains_nested_matmul(child):
                return True
        
        return False
    
    def _generate_loads_and_nested_ops(self, node: ASTNode) -> str:
        """Generate loads and handle nested operations with temporary variables"""
        code = ""
        indent_str = '    ' * self.indent_level
        
        # First, generate all load operations
        code += self._generate_loads_separately(node)
        
        # Then, find and generate nested matmul operations
        nested_matmuls = self._find_nested_matmuls(node)
        for matmul_node in nested_matmuls:
            if matmul_node.children[1].node_type == NodeType.MATMUL:
                # Generate temporary for inner matmul
                inner_matmul = matmul_node.children[1]
                temp_name = f"matmul_temp_{self.temp_counter}"
                self.temp_counter += 1
                
                # Generate the inner matmul computation
                left = inner_matmul.children[0]
                right = inner_matmul.children[1]
                
                if hasattr(left, 'temp_var'):
                    left_expr = left.temp_var
                else:
                    left_expr = f"temp_{self._find_temp_var_index(left)}"
                    
                if hasattr(right, 'temp_var'):
                    right_expr = right.temp_var
                else:
                    right_expr = f"temp_{self._find_temp_var_index(right)}"
                
                code += f"{indent_str}{temp_name} = tl.dot({left_expr}, {right_expr}, allow_tf32=False)\n"
                
                # Mark the node with the temp variable name
                inner_matmul.temp_var = temp_name
        
        return code
    
    def _process_nested_ops(self, node: ASTNode, code_ref):
        """Process nested operations and mark them for temporary variable generation"""
        if node.node_type == NodeType.MATMUL:
            # Check if right child is also a matmul
            if len(node.children) > 1 and node.children[1].node_type == NodeType.MATMUL:
                # This will be handled in _generate_matmul
                pass
        
        # Process children
        for child in node.children:
            if isinstance(child, ASTNode):
                self._process_nested_ops(child, code_ref)
    
    def _get_final_expression(self, node: ASTNode) -> str:
        """Get the final expression after processing nested operations"""
        # Generate without loads, using temporary variables where needed
        return self._generate_node_without_loads(node)
    
    def _walk_tree(self, node: ASTNode):
        """Walk the AST tree and yield all nodes"""
        yield node
        for child in node.children:
            if isinstance(child, ASTNode):
                yield from self._walk_tree(child)
    
    def _find_nested_matmuls(self, node: ASTNode, matmuls=None):
        """Find all matmul nodes that have nested matmuls"""
        if matmuls is None:
            matmuls = []
        
        if node.node_type == NodeType.MATMUL:
            # Check if any child is also a matmul
            for child in node.children:
                if isinstance(child, ASTNode) and child.node_type == NodeType.MATMUL:
                    matmuls.append(node)
                    break
        
        # Recurse into children
        for child in node.children:
            if isinstance(child, ASTNode):
                self._find_nested_matmuls(child, matmuls)
        
        return matmuls
    
    def _find_temp_var_index(self, node: ASTNode) -> int:
        """Find the temp variable index for a load node"""
        if hasattr(node, 'temp_var'):
            # Extract number from temp_N
            return int(node.temp_var.split('_')[-1])
        return 0
    
    def _generate_nested_matmul_temps(self, node: ASTNode) -> str:
        """Generate temporary variables for nested matmul operations"""
        code = ""
        indent_str = '    ' * self.indent_level
        
        # Find matmul nodes with nested matmuls as children
        def find_and_process_matmuls(n):
            if n.node_type == NodeType.MATMUL:
                # Check if right child is also a matmul
                if len(n.children) > 1 and n.children[1].node_type == NodeType.MATMUL:
                    inner_matmul = n.children[1]
                    
                    # Only process if not already processed
                    if not hasattr(inner_matmul, 'temp_var'):
                        # Generate temp variable for inner matmul (A @ B)
                        temp_name = f"matmul_temp_{self.temp_counter}"
                        self.temp_counter += 1
                        
                        # Get operands
                        left_child = inner_matmul.children[0]
                        right_child = inner_matmul.children[1]
                        
                        # Generate expressions for operands
                        left_expr = left_child.temp_var if hasattr(left_child, 'temp_var') else self._generate_node_without_loads(left_child)
                        right_expr = right_child.temp_var if hasattr(right_child, 'temp_var') else self._generate_node_without_loads(right_child)
                        
                        # Generate the computation
                        code_line = f"{indent_str}{temp_name} = tl.dot({left_expr}, {right_expr}, allow_tf32=False)\n"
                        
                        # Store temp_var for later use
                        inner_matmul.temp_var = temp_name
                        
                        return code_line
            
            # Recurse into children
            for child in n.children:
                if isinstance(child, ASTNode):
                    result = find_and_process_matmuls(child)
                    if result:
                        return result
            return ""
        
        # Process the tree
        code += find_and_process_matmuls(node)
        
        return code
    
    def _contains_loads(self, node: ASTNode) -> bool:
        """Check if node contains load operations or operations that need temp vars"""
        if node.node_type in [NodeType.LOAD, NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            return True
        for child in node.children:
            if isinstance(child, ASTNode) and self._contains_loads(child):
                return True
        return False
    
    def _contains_reduce_sum(self, node: ASTNode) -> bool:
        """Check if node contains reduce sum operations"""
        if node.node_type == NodeType.RSUM:
            return True
        for child in node.children:
            if isinstance(child, ASTNode) and self._contains_reduce_sum(child):
                return True
        return False
    
    def _generate_loads_separately(self, node: ASTNode) -> str:
        """Generate load operations and tensor operations separately before using them"""
        code = ""
        if node.node_type == NodeType.LOAD:
            # Generate load without additional indentation
            code += self._generate_node(node) + "\n"
        elif node.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # Generate these operations as well since they produce temp vars
            code += self._generate_node(node)
            if not code.endswith('\n'):
                code += '\n'
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    code += self._generate_loads_separately(child)
        return code
    
    def _generate_node_without_loads(self, node: ASTNode) -> str:
        """Generate node expression using temp variables for loads and tensor ops"""
        if node.node_type == NodeType.LOAD:
            # Return the temp variable assigned during load generation
            return node.temp_var if hasattr(node, 'temp_var') else self._generate_node(node)
        elif node.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # Return the temp variable assigned during operation generation
            return node.temp_var if hasattr(node, 'temp_var') else self._generate_node(node)
        elif node.node_type in [NodeType.ADD, NodeType.SUB, NodeType.MUL, NodeType.DIV]:
            return self._generate_binary_op(node, {NodeType.ADD: '+', NodeType.SUB: '-', NodeType.MUL: '*', NodeType.DIV: '/'}[node.node_type])
        elif node.node_type == NodeType.MATMUL:
            # Handle matmul specially - check if it has a temp_var from nested processing
            if hasattr(node, 'temp_var'):
                return node.temp_var
            else:
                # Generate matmul expression
                left = node.children[0]
                right = node.children[1]
                left_expr = self._generate_node_without_loads(left)
                right_expr = self._generate_node_without_loads(right)
                return f"tl.dot({left_expr}, {right_expr}, allow_tf32=False)"
        elif node.node_type == NodeType.RSUM:
            # Handle reduce sum specially
            child = node.children[0]
            axis = self._generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.sum({child.temp_var}, axis={axis})"
            else:
                child_expr = self._generate_node_without_loads(child)
                return f"tl.sum({child_expr}, axis={axis})"
        elif node.node_type == NodeType.BCAST:
            # Handle broadcast specially
            return self._generate_broadcast(node)
        else:
            return self._generate_node(node)
    
    def _generate_permute3(self, node: ASTNode) -> str:
        """Generate permute3 operation for 3D tensor
        
        permute3 in the IR takes:
        1. The tensor to permute
        2-4. Three dimension indices for the permutation
        
        Example: (permute3 tensor 1 0 2) means permute dimensions as (1, 0, 2)
        """
        if len(node.children) != 4:
            raise ValueError("permute3 requires exactly 4 arguments: tensor, dim0, dim1, dim2")
        
        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""
        
        # If child is a load and doesn't have temp_var yet, generate it
        if child.node_type == NodeType.LOAD and not hasattr(child, 'temp_var'):
            child_code = self._generate_node(child) + "\n"
        
        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            # For simple expressions, evaluate inline
            tensor_expr = self._generate_node(child)
        
        # Get the permutation dimensions
        dim0 = self._generate_node(node.children[1])
        dim1 = self._generate_node(node.children[2])
        dim2 = self._generate_node(node.children[3])
        
        # If used inline (e.g., in a store), return the expression directly
        if hasattr(self, '_generating_inline') and self._generating_inline:
            return f"tl.permute({tensor_expr}, ({dim0}, {dim1}, {dim2}))"
        
        # Generate a temporary variable for the result
        temp_var = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        # In Triton, use tl.permute
        indent = '    ' * self.indent_level
        code = child_code
        code += f"{indent}{temp_var} = tl.permute({tensor_expr}, ({dim0}, {dim1}, {dim2}))\n"
        
        # Store temp var in node for parent operations
        node.temp_var = temp_var
        
        return code
    
    def _generate_squeeze(self, node: ASTNode) -> str:
        """Generate squeeze operation to remove a dimension
        
        squeeze in the IR takes:
        1. The tensor to squeeze
        2. The dimension to remove
        
        Example: (squeeze tensor 1) means remove dimension 1
        """
        if len(node.children) != 2:
            raise ValueError("squeeze requires exactly 2 arguments: tensor, dim")
        
        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""
        
        # If child needs generation, do it first
        if child.node_type in [NodeType.LOAD, NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE] and not hasattr(child, 'temp_var'):
            child_code = self._generate_node(child)
            if not child_code.endswith('\n'):
                child_code += '\n'
        
        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            # For simple expressions, evaluate inline
            tensor_expr = self._generate_node(child)
        
        # Get the dimension to squeeze
        dim = self._generate_node(node.children[1])
        
        # Generate a temporary variable for the result
        temp_var = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        # Try to infer the source tensor name for dimension information
        source_tensor_name = self._infer_tensor_name(child)
        
        indent = '    ' * self.indent_level
        code = child_code
        
        if source_tensor_name:
            # Use tensor dimension information to construct new shape for reshape
            # Build new shape by excluding the squeezed dimension
            code += f"{indent}# Squeeze dimension {dim} from {source_tensor_name}\n"
            
            # Get the number of dimensions from tensor_shapes if available
            if source_tensor_name in self.tensor_shapes:
                num_dims = len(self.tensor_shapes[source_tensor_name])
            else:
                # Default to 3D for typical attention tensors
                num_dims = 3
            
            shape_parts = []
            for i in range(num_dims):
                if i != int(dim):  # Skip the dimension to be squeezed
                    shape_parts.append(f"{source_tensor_name}_dim{i}")
            
            # Pass shape directly as tuple to tl.reshape
            shape_tuple = f"({', '.join(shape_parts)})"
            code += f"{indent}{temp_var} = tl.reshape({tensor_expr}, {shape_tuple})\n"
        else:
            # Fallback: simple assignment with comment
            code += f"{indent}{temp_var} = {tensor_expr}  # squeeze dim {dim} (fallback)\n"
        
        # Store temp var in node for parent operations
        node.temp_var = temp_var
        
        return code
    
    def _infer_tensor_name(self, node: ASTNode) -> str:
        """Infer the original tensor name from a node (for accessing dimension info)"""
        if node.node_type == NodeType.LOAD:
            # Direct load operation
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                return tensor_node.children[0].value
        elif node.node_type == NodeType.PERMUTE3:
            # Permute operation - infer from its child
            return self._infer_tensor_name(node.children[0])
        elif node.node_type == NodeType.UNSQUEEZE:
            # Unsqueeze operation - infer from its child
            return self._infer_tensor_name(node.children[0])
        elif hasattr(node, 'temp_var') and hasattr(node, 'original_tensor'):
            # If we stored the original tensor name
            return node.original_tensor
        
        return None
    
    def _generate_unsqueeze(self, node: ASTNode) -> str:
        """Generate unsqueeze operation to add a dimension
        
        unsqueeze in the IR takes:
        1. The tensor to unsqueeze
        2. The dimension where to insert new dimension
        
        Example: (unsqueeze tensor 1) means add a dimension at position 1
        """
        if len(node.children) != 2:
            raise ValueError("unsqueeze requires exactly 2 arguments: tensor, dim")
        
        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""
        
        # If child is a load and doesn't have temp_var yet, generate it
        if child.node_type == NodeType.LOAD and not hasattr(child, 'temp_var'):
            child_code = self._generate_node(child) + "\n"
        
        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            # For simple expressions, evaluate inline
            tensor_expr = self._generate_node(child)
        
        # Get the dimension to unsqueeze
        dim = self._generate_node(node.children[1])
        
        # Generate a temporary variable for the result
        temp_var = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        # In Triton, use tl.expand_dims
        indent = '    ' * self.indent_level
        code = child_code
        code += f"{indent}{temp_var} = tl.expand_dims({tensor_expr}, {dim})\n"
        
        # Store temp var in node for parent operations
        node.temp_var = temp_var
        
        return code