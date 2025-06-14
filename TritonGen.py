from typing import Dict, Tuple
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
        self.program_id_counter = 0
        self.tensors_used = set()
        self.offset_counter = 0  # Counter for offset variables
        self.mask_counter = 0    # Counter for mask variables
        self.loop_var_to_tensor_dim = {}  # loop_var -> (tensor_name, dimension)
        
    def generate(self, ast: ASTNode, tensor_shapes: Dict[str, Tuple[int, ...]] = None) -> str:
        """Generate Triton kernel code from AST"""
        if tensor_shapes:
            self.tensor_shapes = tensor_shapes
            
        # Reset state
        self.parallel_dims = []
        self.program_id_counter = 0
        self.tensors_used = set()
        self.temp_counter = 0
        self.offset_counter = 0
        self.mask_counter = 0
        self.loop_var_to_tensor_dim = {}
        
        # First pass: collect all tensors used in the AST
        self._collect_tensors(ast)
        
        # Second pass: collect parallel loops to determine BLOCK parameters
        self._collect_parallel_loops(ast)
        
        # Third pass: analyze loop contexts to map loop vars to tensor dimensions
        self._analyze_loop_contexts(ast)

        # Generate kernel function
        kernel_code = self._generate_kernel_header()

        self.indent_level = 1  # Set initial indent level for kernel body
        kernel_code += self._generate_node(ast)
        self.indent_level = 0  # Reset indent level
        
        return kernel_code
    
    def _collect_tensors(self, node: ASTNode):
        """Collect all tensor names used in the AST"""
        if node.node_type == NodeType.INPUT:
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
    
    def _generate_kernel_header(self) -> str:
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
        # block sizes for each parallel loop
        for loop_var, start, end, tile_size in self.parallel_dims:
            params.append(f"BLOCK_{loop_var.upper()}: tl.constexpr")
        params_str = ",\n    ".join(params)
        header = f"""import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    {params_str}
):
"""
        return header

#     def _generate_kernel_header(self) -> str:
#         """Generate Triton kernel function header"""
#         return """import triton
# import triton.language as tl
# import torch

# @triton.jit
# def kernel(
#     {params_str},
# ):
    
# """
    
#     def _generate_launcher(self) -> str:
#         """Generate kernel launcher function"""
#         return """def launch_kernel(tensors, grid_size=None, block_size=128):
#     \"\"\"Launch the generated Triton kernel\"\"\"
#     # Extract tensor pointers and metadata
#     tensor_ptrs = {}
#     shapes_and_strides = {}
    
#     for name, tensor in tensors.items():
#         tensor_ptrs[f'{name}_ptr'] = tensor.data_ptr()
#         shapes_and_strides[f'{name}_shape'] = tensor.shape
#         shapes_and_strides[f'{name}_stride'] = tensor.stride()
    
#     # Auto-calculate grid size based on parallel dimensions if not provided
#     if grid_size is None:
#         # This should be set based on the parallel loops found during code generation
#         # For now, use a default that works for most cases
#         grid_size = (1,)
    
#     # Launch kernel
#     kernel[grid_size](
#         **tensor_ptrs,
#         **shapes_and_strides,
#         BLOCK_SIZE=block_size,
#     )
# """
    
    def _generate_node(self, node: ASTNode) -> str:
        """Generate code for a single AST node"""
        if node.node_type == NodeType.PLOOP:
            return self._generate_ploop(node)
        elif node.node_type == NodeType.SLOOP:
            return self._generate_sloop(node)
        elif node.node_type == NodeType.SEQ:
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
        elif node.node_type == NodeType.NUM:
            return str(node.value)
        elif node.node_type == NodeType.VAR:
            return node.value
        else:
            return f"# TODO: Implement {node.node_type.value}"
    
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
        """Generate sequence of operations"""
        code = ""
        for child in node.children:
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
        if self._contains_loads(val_node) or self._contains_reduce_sum(val_node):
            code += self._generate_loads_separately(val_node)
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
        
        # Generate multi-dimensional indexing
        if len(offsets) == 1:
            # For 1D tensors, still need to multiply by stride
            offset_str = f"({offsets[0]}) * {tensor_name}_stride0"
            mask_str = masks[0]
        else:
            # Multi-dimensional indexing: i[:,None]*tensor_stride0 + j[None,:]*tensor_stride1
            offset_parts = []
            mask_parts = []
            
            for dim, off in enumerate(offsets):
                if dim == 0:
                    # First dimension: i[:,None]
                    offset_parts.append(f"({off})[:, None] * {tensor_name}_stride{dim}")
                    mask_parts.append(f"({masks[dim]})[:, None]")
                elif dim == 1:
                    # Second dimension: j[None,:]
                    offset_parts.append(f"({off})[None, :] * {tensor_name}_stride{dim}")
                    mask_parts.append(f"({masks[dim]})[None, :]")
                else:
                    # Higher dimensions - extend with more None dimensions
                    none_before = ", ".join(["None"] * dim)
                    none_after = ", ".join(["None"] * (len(offsets) - dim - 1))
                    if none_before and none_after:
                        offset_parts.append(f"({off})[{none_before}, :, {none_after}] * {tensor_name}_stride{dim}")
                        mask_parts.append(f"({masks[dim]})[{none_before}, :, {none_after}]")
                    elif none_before:
                        offset_parts.append(f"({off})[{none_before}, :] * {tensor_name}_stride{dim}")
                        mask_parts.append(f"({masks[dim]})[{none_before}, :]")
                    elif none_after:
                        offset_parts.append(f"({off})[:, {none_after}] * {tensor_name}_stride{dim}")
                        mask_parts.append(f"({masks[dim]})[:, {none_after}]")
                    else:
                        offset_parts.append(f"({off}) * {tensor_name}_stride{dim}")
                        mask_parts.append(f"{masks[dim]}")
            
            offset_str = " + ".join(offset_parts)
            # Create combined mask using multiplication instead of &
            mask_str = " * ".join(mask_parts)
        
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
        
        # Use temp_var if available (from load operations), otherwise generate normally
        if hasattr(left_child, 'temp_var'):
            left = left_child.temp_var
        else:
            left = self._generate_node(left_child)
            
        if hasattr(right_child, 'temp_var'):
            right = right_child.temp_var
        else:
            right = self._generate_node(right_child)
            
        # For DNN compiler, use tl.dot with default allow_tf32=True
        # The IR already handles tiling, so we just use tl.dot on the tiles
        return f"tl.dot({left}, {right})"
    
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
            
            if tensor_node.node_type == NodeType.INPUT:
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
    
    def _collect_parallel_loops(self, node: ASTNode):
        """Collect parallel loop information for BLOCK parameters"""
        if node.node_type == NodeType.PLOOP:
            # Extract loop info: (ploop start end tile_size loop_var body)
            start = node.children[0].value if node.children[0].node_type == NodeType.NUM else "0"
            end = node.children[1].value if node.children[1].node_type == NodeType.NUM else "N"
            tile_size = node.children[2].value if node.children[2].node_type == NodeType.NUM else "BLOCK"
            loop_var = node.children[3].value
            
            # Add to parallel dimensions
            self.parallel_dims.append((loop_var, str(start), str(end), str(tile_size)))
            
            # Continue collecting in body
            if len(node.children) > 4:
                self._collect_parallel_loops(node.children[4])
        elif node.node_type == NodeType.SLOOP:
            # Also collect sequential loops for BLOCK parameters
            start = node.children[0].value if node.children[0].node_type == NodeType.NUM else "0"
            end = node.children[1].value if node.children[1].node_type == NodeType.NUM else "N"
            tile_size = node.children[2].value if node.children[2].node_type == NodeType.NUM else "BLOCK"
            loop_var = node.children[3].value
            
            # Add to parallel dimensions (will be used for BLOCK parameters)
            self.parallel_dims.append((loop_var, str(start), str(end), str(tile_size)))
            
            # Continue collecting in body
            if len(node.children) > 4:
                self._collect_parallel_loops(node.children[4])
        else:
            # Recursively collect from children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._collect_parallel_loops(child)
    
    def _contains_loads(self, node: ASTNode) -> bool:
        """Check if node contains load operations"""
        if node.node_type == NodeType.LOAD:
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
        """Generate load operations separately before using them"""
        code = ""
        if node.node_type == NodeType.LOAD:
            # Generate load without additional indentation
            code += self._generate_node(node) + "\n"
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    code += self._generate_loads_separately(child)
        return code
    
    def _generate_node_without_loads(self, node: ASTNode) -> str:
        """Generate node expression using temp variables for loads"""
        if node.node_type == NodeType.LOAD:
            # Return the temp variable assigned during load generation
            return node.temp_var if hasattr(node, 'temp_var') else self._generate_node(node)
        elif node.node_type in [NodeType.ADD, NodeType.SUB, NodeType.MUL, NodeType.DIV]:
            return self._generate_binary_op(node, {NodeType.ADD: '+', NodeType.SUB: '-', NodeType.MUL: '*', NodeType.DIV: '/'}[node.node_type])
        elif node.node_type == NodeType.RSUM:
            # Handle reduce sum specially
            child = node.children[0]
            axis = self._generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.sum({child.temp_var}, axis={axis})"
            else:
                child_expr = self._generate_node_without_loads(child)
                return f"tl.sum({child_expr}, axis={axis})"
        else:
            return self._generate_node(node)