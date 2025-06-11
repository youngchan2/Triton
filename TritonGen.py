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
        
    def generate(self, ast: ASTNode, tensor_shapes: Dict[str, Tuple[int, ...]] = None) -> str:
        """Generate Triton kernel code from AST"""
        if tensor_shapes:
            self.tensor_shapes = tensor_shapes
            
        # Reset state
        self.parallel_dims = []
        self.program_id_counter = 0
        self.tensors_used = set()
        
        # First pass: collect all tensors used in the AST
        self._collect_tensors(ast)

        # Generate kernel function
        kernel_code = self._generate_kernel_header()
        kernel_code += self._generate_node(ast)
        kernel_code += self._generate_kernel_footer()
        
        # Generate launcher function with grid size calculation
        launcher_code = self._generate_launcher()
        
        return kernel_code + "\n\n" + launcher_code
    
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
        """Generate Triton kernel function header"""
        return """import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    {params_str},
):
    
"""
    
    def _generate_kernel_footer(self) -> str:
        """Generate kernel footer"""
        return ""
    
    def _generate_launcher(self) -> str:
        """Generate kernel launcher function"""
        return """def launch_kernel(tensors, grid_size=None, block_size=128):
    \"\"\"Launch the generated Triton kernel\"\"\"
    # Extract tensor pointers and metadata
    tensor_ptrs = {}
    shapes_and_strides = {}
    
    for name, tensor in tensors.items():
        tensor_ptrs[f'{name}_ptr'] = tensor.data_ptr()
        shapes_and_strides[f'{name}_shape'] = tensor.shape
        shapes_and_strides[f'{name}_stride'] = tensor.stride()
    
    # Auto-calculate grid size based on parallel dimensions if not provided
    if grid_size is None:
        # This should be set based on the parallel loops found during code generation
        # For now, use a default that works for most cases
        grid_size = (1,)
    
    # Launch kernel
    kernel[grid_size](
        **tensor_ptrs,
        **shapes_and_strides,
        BLOCK_SIZE=block_size,
    )
"""
    
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
        # Each ploop becomes a grid dimension
        grid_dim = len(self.parallel_dims)
        self.parallel_dims.append((loop_var, start, end, tile_size))
        
        code = f"""    # Parallel loop {loop_var} from {start} to {end} with tile size {tile_size}
    # Executed across grid dimension {grid_dim}
    {loop_var} = {start} + tl.program_id({grid_dim}) * {tile_size}
    
    # Boundary check for parallel execution
    if {loop_var} < {end}:
"""
        self.indent_level += 1
        code += self._indent(self._generate_node(body))
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
        
        code = f"""    # Sequential loop {loop_var} from {start} to {end} with tile size {tile_size}
    for {loop_var} in range({start}, {end}, {tile_size}):
"""
        self.indent_level += 1
        code += self._indent(self._generate_node(body))
        self.indent_level -= 1
        
        return code
    
    def _generate_seq(self, node: ASTNode) -> str:
        """Generate sequence of operations"""
        code = ""
        for child in node.children:
            code += self._indent(self._generate_node(child)) + "\n"
        return code
    
    def _generate_load(self, node: ASTNode) -> str:
        """Generate load operation"""
        # (load tensor index)
        tensor_node = node.children[0]
        index_node = node.children[1]
        
        tensor_name = tensor_node.children[0].value
        offsets, mask = self._generate_index(index_node)
        
        temp_var = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        return f"{temp_var} = tl.load({tensor_name}_ptr + {offsets}, mask={mask})"
    
    def _generate_store(self, node: ASTNode) -> str:
        """Generate store operation"""
        # (store tensor val index)
        tensor_node = node.children[0]
        val_node = node.children[1]
        index_node = node.children[2]
        
        tensor_name = tensor_node.children[0].value
        value_expr = self._generate_node(val_node)
        offsets, mask = self._generate_index(index_node)
        
        return f"tl.store({tensor_name}_ptr + {offsets}, {value_expr}, mask={mask})"
    
    def _generate_index(self, node: ASTNode) -> Tuple[str, str]:
        """Generate index calculation for memory access"""
        # (index tile1 tile2 ...)
        offsets = []
        masks = []
        
        for i, child in enumerate(node.children):
            if child.node_type == NodeType.TILE:
                loop_var = child.children[0].value
                if loop_var in self.loop_vars:
                    start, end, tile_size, is_parallel = self.loop_vars[loop_var]
                    
                    if is_parallel:
                        # For parallel loops, create tile range from program_id * tile_size
                        offset_expr = f"{loop_var} + tl.arange(0, {tile_size})"
                        mask_expr = f"({loop_var} + tl.arange(0, {tile_size})) < {end}"
                    else:
                        # For sequential loops, use regular indexing
                        offset_expr = f"{loop_var} + tl.arange(0, {tile_size})"
                        mask_expr = f"({loop_var} + tl.arange(0, {tile_size})) < {end}"
                    
                    offsets.append(offset_expr)
                    masks.append(mask_expr)
            elif child.node_type == NodeType.FULLTILE:
                offsets.append("tl.arange(0, BLOCK_SIZE)")
                masks.append("True")
        
        # Generate multi-dimensional indexing
        if len(offsets) == 1:
            offset_str = offsets[0]
            mask_str = masks[0]
        else:
            # Multi-dimensional indexing: i[:,None]*stride_0 + j[None,:]*stride_1
            offset_parts = []
            for dim, off in enumerate(offsets):
                if dim == 0:
                    # First dimension: i[:,None]
                    offset_parts.append(f"({off})[:, None] * stride_{dim}")
                elif dim == 1:
                    # Second dimension: j[None,:]
                    offset_parts.append(f"({off})[None, :] * stride_{dim}")
                else:
                    # Higher dimensions - extend with more None dimensions
                    none_before = ", ".join(["None"] * dim)
                    none_after = ", ".join(["None"] * (len(offsets) - dim - 1))
                    if none_before and none_after:
                        offset_parts.append(f"({off})[{none_before}, :, {none_after}] * stride_{dim}")
                    elif none_before:
                        offset_parts.append(f"({off})[{none_before}, :] * stride_{dim}")
                    elif none_after:
                        offset_parts.append(f"({off})[:, {none_after}] * stride_{dim}")
                    else:
                        offset_parts.append(f"({off}) * stride_{dim}")
            
            offset_str = " + ".join(offset_parts)
            mask_str = " & ".join(masks)
        
        return offset_str, mask_str
    
    def _generate_binary_op(self, node: ASTNode, op: str) -> str:
        """Generate binary operation"""
        left = self._generate_node(node.children[0])
        right = self._generate_node(node.children[1])
        return f"({left} {op} {right})"
    
    def _generate_unary_op(self, node: ASTNode, op: str) -> str:
        """Generate unary operation"""
        operand = self._generate_node(node.children[0])
        return f"{op}({operand})"
    
    def _generate_matmul(self, node: ASTNode) -> str:
        """Generate matrix multiplication"""
        left = self._generate_node(node.children[0])
        right = self._generate_node(node.children[1])
        return f"tl.dot({left}, {right})"
    
    def _generate_reduce_sum(self, node: ASTNode) -> str:
        """Generate reduce sum operation"""
        tensor = self._generate_node(node.children[0])
        axis = self._generate_node(node.children[1])
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