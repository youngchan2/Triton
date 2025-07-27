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
        self.loop_var_to_tensor_dim = {}  # loop_var -> (tensor_name, dimension)
        self.kernel_counter = 0  # Counter for kernel names
        self.intermediate_tensors = set()  # Track intermediate tensors (input tensors used in store)
        self.generated_kernels = []  # List of (kernel_name, kernel_code, tensors_used, parallel_dims)
        self.cross_kernel_tensors = set()  # Track tensors that cross kernel boundaries
        self.stored_tensor_dims = {}  # Track which tensors are stored in parallel loops: loop_var -> [(tensor, dim)]
        self.load_cache = {}  # Cache for load operations: (tensor_name, offset_expr) -> temp_var
        self.constants = {}  # Map variable names (e.g., 'N', 'M') to their constant values
        self.sloop_temp_vars = {}  # Map tensor_name -> temp_var_name for sloop non-accumulation stores
        self.current_sloop_info = None  # Track current sloop context: (loop_var, sloop_node)
        
    def resolve_value(self, value) -> str:
        """Resolve a value that might be a variable name or expression to its constant value.
        
        Args:
            value: Could be a string, int, or other type
            
        Returns:
            The resolved value as a string
        """
        # If it's already a number (int or float), return as string
        if isinstance(value, (int, float)):
            return str(value)
            
        # Convert to string if not already
        value = str(value)
        
        # If it's already a number string, return as is
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return value
            
        # Check if it's a simple variable in our constants dictionary
        if value in self.constants:
            return str(self.constants[value])
            
        # Check if it contains an expression (e.g., P+M, N-1, etc.)
        if any(op in value for op in ['+', '-', '*', '//']):
            # Try to evaluate the expression by replacing variables with their values
            expr = value
            for var_name, var_value in self.constants.items():
                # Replace whole words only (to avoid replacing 'M' in 'MAX')
                import re
                expr = re.sub(r'\b' + var_name + r'\b', str(var_value), expr)
            
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
                
        # Otherwise return the original value
        return value
    
    def resolve_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Resolve symbolic tensor shapes to concrete values using constants.
        
        Returns:
            Dictionary mapping tensor names to resolved shape tuples
        """
        resolved_shapes = {}
        
        for tensor_name, shape in self.tensor_shapes.items():
            resolved_shape = []
            for dim in shape:
                if isinstance(dim, str):
                    # It's a symbolic dimension or expression, resolve it
                    resolved_dim = self.resolve_value(dim)
                    try:
                        # Convert to integer
                        resolved_shape.append(int(resolved_dim))
                    except ValueError:
                        raise ValueError(f"Could not resolve dimension '{dim}' to an integer value")
                else:
                    # It's already a number
                    resolved_shape.append(dim)
            resolved_shapes[tensor_name] = tuple(resolved_shape)
            
        return resolved_shapes
        
    def generate(self, ast: ASTNode, tensor_shapes: Dict[str, Tuple[int, ...]] = None, constants: Dict[str, int] = None) -> str:
        """Generate Triton kernel code from AST
        
        Args:
            ast: The AST to generate code from
            tensor_shapes: Optional tensor shape information
            constants: Optional mapping of variable names to constant values
        """
        # Store current AST for use in sloop generation
        self.current_ast = ast
        
        if tensor_shapes:
            self.tensor_shapes = tensor_shapes
        if constants:
            self.constants = constants
            
        # Reset state
        self.kernel_counter = 0
        self.generated_kernels = []
        self.cross_kernel_tensors = set()
        
        # Check if root is a seq node
        if ast.node_type == NodeType.SEQ:
            # First, identify cross-kernel tensors
            self._identify_cross_kernel_tensors(ast)
            # Generate separate kernels for each child
            return self._generate_seq_kernels(ast)
        else:
            # Generate single kernel with wrapper function
            all_code = ""
            
            # Generate import statements once
            all_code += """import triton
import triton.language as tl
import torch

"""
            
            # Generate the single kernel
            kernel_name = "kernel_0"
            kernel_code = self._generate_single_kernel(ast, kernel_name=kernel_name)
            
            # Remove the import statements from kernel (they're already at the top)
            kernel_code = kernel_code.replace("import triton\nimport triton.language as tl\nimport torch\n\n", "")
            
            all_code += kernel_code + "\n\n"
            
            # Store kernel info for wrapper generation
            self.generated_kernels.append((
                kernel_name,
                kernel_code,
                self.tensors_used.copy(),
                self.parallel_dims.copy(),
                self.all_loops.copy(),
                self.stored_tensor_dims.copy()
            ))
            
            # Generate wrapper function
            all_code += self._generate_wrapper_function()
            
            return all_code
    
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
                self.all_loops.copy(),
                self.stored_tensor_dims.copy()
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
        
        for _, _, tensors_used, _, all_loops, _ in self.generated_kernels:
            all_tensors.update(tensors_used)
            # Collect block parameters from loops
            for loop_var, _, _, tile_size in all_loops:
                # Check if tile_size is a variable (not a number)
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    block_param = f"block_{loop_var}"
                    all_block_params.add(block_param)
        
        # Filter tensors to only include:
        # 1. Output tensors
        # 2. Input tensors that appear in load operations (not intermediate)
        # This excludes intermediate tensors that are not cross-kernel
        forward_params = []
        for tensor_name in sorted(all_tensors):
            # Include if it's not an intermediate tensor, or if it's a cross-kernel tensor
            if tensor_name not in self.intermediate_tensors or tensor_name in self.cross_kernel_tensors:
                forward_params.append(tensor_name)
        
        # Generate metadata for benchmark.py
        code = f"# Metadata for benchmark.py\n"
        code += f"TENSOR_PARAMS = {forward_params}\n"
        code += f"BLOCK_PARAMS = {sorted(list(all_block_params))}\n\n"
        
        code += "def forward("
        
        # Generate parameter list
        params = []
        for tensor_name in forward_params:
            params.append(f"{tensor_name}")
        
        # Add block size parameters with defaults
        for block_param in sorted(all_block_params):
            params.append(f"{block_param}=16")
        
        code += ", ".join(params) + "):\n"
        code += '    """\n'
        code += '    Wrapper function that executes all kernels sequentially.\n'
        code += '    """\n'
        
        # Call each kernel
        for kernel_name, _, tensors_used, parallel_dims, all_loops, stored_tensor_dims in self.generated_kernels:
            # Calculate grid size for this kernel
            if parallel_dims:
                grid_parts = []
                for idx, (loop_var, start, end, tile_size) in enumerate(parallel_dims):
                    # Try to use tensor dimensions for end values
                    end_param = end
                    
                    # Check if it's a variable like M, N, K
                    if end in ['M', 'N', 'K'] or (isinstance(end, str) and end.isalpha()):
                        # First, check stored_tensor_dims for tensors stored with this loop var
                        if loop_var in stored_tensor_dims:
                            # Find the tensor with the largest dimension that uses this loop var
                            max_size = 0
                            best_tensor = None
                            best_dim = None
                            
                            for tensor_name, dim in stored_tensor_dims[loop_var]:
                                if tensor_name in tensors_used and tensor_name in self.tensor_shapes:
                                    shape = self.tensor_shapes[tensor_name]
                                    if dim < len(shape) and shape[dim] > max_size:
                                        max_size = shape[dim]
                                        best_tensor = tensor_name
                                        best_dim = dim
                            
                            if best_tensor:
                                end_param = f"{best_tensor}.shape[{best_dim}]"
                        
                        # If not found in stored_tensor_dims, try loop_var_to_tensor_dim
                        if end_param == end and loop_var in self.loop_var_to_tensor_dim:
                            tensor_name, dim = self.loop_var_to_tensor_dim[loop_var]
                            if tensor_name in tensors_used:
                                end_param = f"{tensor_name}.shape[{dim}]"
                        
                        # If still not resolved, fall back to heuristic
                        if end_param == end:  # Still unchanged
                            # Look for a 2D tensor to get dimensions from
                            found = False
                            for tensor in sorted(tensors_used):
                                if tensor in self.tensor_shapes:
                                    shape = self.tensor_shapes[tensor]
                                    if len(shape) >= 2:
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
            
            # First, add parameters for non-intermediate tensors (in sorted order)
            non_intermediate_tensors = sorted([t for t in tensors_used if t not in self.intermediate_tensors or t in self.cross_kernel_tensors])
            for tensor in non_intermediate_tensors:
                # Pointer
                kernel_params.append(f"        {tensor}")
                
                # NO shape parameters - using constants instead
                
                # Stride parameters
                if tensor in self.tensor_shapes:
                    num_dims = len(self.tensor_shapes[tensor])
                else:
                    num_dims = max(len(parallel_dims), 2)  # Default to 2D
                
                for dim in range(num_dims):
                    kernel_params.append(f"        {tensor}.stride({dim})")
            
            # NO shape parameters for local intermediate tensors - using constants instead
            
            # Add block size parameters for all loops
            for loop_var, _, _, tile_size in all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    # Use block parameter variable
                    kernel_params.append(f"        block_{loop_var}")
                else:
                    # Use numeric value
                    kernel_params.append(f"        {tile_size}")
            
            # Add constant values
            for const_name in sorted(self.constants.keys()):
                kernel_params.append(f"        {self.constants[const_name]}")
            
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
        self.loop_var_to_tensor_dim = {}
        self.loop_vars = {}
        self.stored_tensor_dims = {}
        self.load_cache = {}  # Reset load cache for each kernel
        self.intermediate_tensor_indices = {}  # Reset intermediate tensor indices
        self.sloop_temp_vars = {}  # Clear sloop temp vars for new kernel
        self.current_sloop_info = None  # Reset sloop context
        
        # First pass: collect all tensors used in the AST
        self._collect_tensors(ast)
        
        # Second pass: collect intermediate tensors (input tensors in store operations)
        self._collect_intermediate_tensors(ast, in_ploop=False, ploop_var=None)
        
        # Third pass: collect all loops to determine BLOCK parameters
        self._collect_all_loops(ast)
        
        # Fourth pass: analyze loop contexts to map loop vars to tensor dimensions
        self._analyze_loop_contexts(ast)
        
        # Fifth pass: analyze which tensors are stored in parallel loops
        self._analyze_parallel_store_dims(ast)

        # Generate kernel function
        kernel_code = self._generate_kernel_header(kernel_name)

        self.indent_level = 1  # Set initial indent level for kernel body
        
        # Constants are now passed as parameters, no need to define them
        # kernel_code += self._generate_constants_definition()
        
        # Generate intermediate tensor allocations
        kernel_code += self._generate_intermediate_allocations(ast)
        
        kernel_code += self._generate_node(ast)
        self.indent_level = 0  # Reset indent level
        
        return kernel_code
    
    def _collect_tensors(self, node: ASTNode):
        """Collect all tensor names used in the AST"""
        if node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
            # Handle multiple tensor names (comma-separated)
            for child in node.children:
                if child.node_type == NodeType.VAR:
                    tensor_name = child.value
                    self.tensors_used.add(tensor_name)

        for child in node.children:
            if isinstance(child, ASTNode):
                self._collect_tensors(child)
    
    def _collect_intermediate_tensors(self, node: ASTNode, in_ploop=False, ploop_var=None):
        """Collect intermediate tensors (tensor operators used in store operations)"""
        if node.node_type == NodeType.STORE:
            tensor_node = node.children[0]
            if tensor_node.node_type == NodeType.TENSOR:
                # Handle multiple tensor names (comma-separated)
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        tensor_name = child.value

                        # Tensor operators represent intermediate tensors
                        # These need to be initialized with tl.zeros
                        self.intermediate_tensors.add(tensor_name)
                        
                        # Infer shape from the store's index using existing tensor shapes
                        if tensor_name not in self.tensor_shapes and len(node.children) >= 3:
                            index_node = node.children[2]
                            shape = self._infer_tensor_shape_from_index(index_node)
                            if shape:
                                self.tensor_shapes[tensor_name] = shape
                        
                        # Analyze the index pattern to determine actual shape needed
                        if len(node.children) >= 3:
                            index_node = node.children[2]
                            if index_node.node_type == NodeType.INDEX:
                                # Store the index pattern for this tensor
                                if not hasattr(self, 'intermediate_tensor_indices'):
                                    self.intermediate_tensor_indices = {}
                                self.intermediate_tensor_indices[tensor_name] = index_node
        
        # Handle ploop nodes
        if node.node_type == NodeType.PLOOP:
            # Extract ploop variable
            loop_var = node.children[3].value if len(node.children) > 3 else None
            # Process children with ploop context
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._collect_intermediate_tensors(child, in_ploop=True, ploop_var=loop_var)
        else:
            # Process children with current context
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._collect_intermediate_tensors(child, in_ploop, ploop_var)
    
    def _uses_ploop_tile_index(self, index_node: ASTNode, ploop_var: str) -> bool:
        """Check if an index uses tile with the ploop variable"""
        if index_node.node_type == NodeType.INDEX:
            # Check each dimension in the index
            for child in index_node.children:
                if child.node_type == NodeType.TILE:
                    # Check if this tile uses the ploop variable
                    if len(child.children) > 0 and child.children[0].value == ploop_var:
                        return True
        return False
    
    def _identify_cross_kernel_tensors(self, seq_node: ASTNode):
        """Identify tensors that cross kernel boundaries in seq operations"""
        if seq_node.node_type != NodeType.SEQ:
            return
        
        # Flatten nested seq nodes to get all operations
        operations = self._flatten_seq(seq_node)
        
        # Track tensor writes and reads per kernel
        tensor_writes = {}  # tensor -> kernel_id where it's written
        tensor_reads = {}   # tensor -> set of kernel_ids where it's read
        
        # Analyze each kernel (operation)
        for kernel_id, op in enumerate(operations):
            # Collect tensor usage in this kernel
            writes_in_kernel = set()
            reads_in_kernel = set()
            self._collect_tensor_usage(op, writes_in_kernel, reads_in_kernel)
            
            # Update global tracking
            for tensor in writes_in_kernel:
                tensor_writes[tensor] = kernel_id
            
            for tensor in reads_in_kernel:
                if tensor not in tensor_reads:
                    tensor_reads[tensor] = set()
                tensor_reads[tensor].add(kernel_id)
        
        # Identify cross-kernel tensors
        for tensor, write_kernel in tensor_writes.items():
            if tensor in tensor_reads:
                for read_kernel in tensor_reads[tensor]:
                    if write_kernel != read_kernel:
                        # This tensor is written in one kernel and read in another
                        self.cross_kernel_tensors.add(tensor)
                        break
    
    def _collect_tensor_usage(self, node: ASTNode, writes: set, reads: set):
        """Collect tensor reads and writes in a node"""
        if node.node_type == NodeType.STORE:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                tensor_name = tensor_node.children[0].value
                writes.add(tensor_name)
            # Also check for reads in the value expression
            if len(node.children) > 1:
                self._collect_tensor_usage(node.children[1], writes, reads)
        elif node.node_type == NodeType.LOAD:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                tensor_name = tensor_node.children[0].value
                reads.add(tensor_name)
        else:
            # Recursively check children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._collect_tensor_usage(child, writes, reads)
    
    def _analyze_parallel_store_dims(self, node: ASTNode, current_ploop_var: str = None):
        """Analyze which tensors are stored in parallel loops and track their dimensions"""
        if node.node_type == NodeType.PLOOP:
            # Extract ploop variable
            loop_var = node.children[3].value
            # Recursively analyze body with this ploop variable
            body = node.children[4]
            self._analyze_parallel_store_dims(body, loop_var)
        elif node.node_type == NodeType.STORE and current_ploop_var:
            # Found a store inside a ploop
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                tensor_name = tensor_node.children[0].value
                # Analyze the index to find which dimension uses the ploop variable
                if len(node.children) >= 3:
                    index_node = node.children[2]
                    dims_using_ploop = self._find_dims_using_loop_var(index_node, current_ploop_var)
                    
                    if current_ploop_var not in self.stored_tensor_dims:
                        self.stored_tensor_dims[current_ploop_var] = []
                    
                    for dim in dims_using_ploop:
                        self.stored_tensor_dims[current_ploop_var].append((tensor_name, dim))
        else:
            # Recursively analyze children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._analyze_parallel_store_dims(child, current_ploop_var)
    
    def _find_dims_using_loop_var(self, index_node: ASTNode, loop_var: str, current_dim: int = 0) -> List[int]:
        """Find which dimensions in an index expression use the given loop variable"""
        dims = []
        
        if index_node.node_type == NodeType.INDEX:
            # Process each dimension
            for i, child in enumerate(index_node.children):
                if self._index_uses_loop_var(child, loop_var):
                    dims.append(i)
        
        return dims
    
    def _index_uses_loop_var(self, node: ASTNode, loop_var: str) -> bool:
        """Check if an index expression uses the given loop variable"""
        if node.node_type == NodeType.TILE:
            # (tile loop_var)
            if node.children and node.children[0].value == loop_var:
                return True
        elif node.node_type == NodeType.ELEM:
            # (elem loop_var)
            if node.children and node.children[0].value == loop_var:
                return True
        
        # Recursively check children
        for child in node.children:
            if isinstance(child, ASTNode):
                if self._index_uses_loop_var(child, loop_var):
                    return True
        
        return False
    
    def _identify_accumulators(self, ast: ASTNode) -> set:
        """Identify which intermediate tensors are accumulators"""
        accumulators = set()
        
        def analyze_store(node: ASTNode, in_sloop: bool = False):
            """Check if a store operation is an accumulation"""
            if node.node_type == NodeType.STORE and len(node.children) >= 2:
                # Get the tensor being stored to
                tensor_node = node.children[0]
                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    # Handle multiple tensor names (comma-separated)
                    for i, child in enumerate(tensor_node.children):
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            
                            # For comma-separated tensors, need to check the corresponding load
                            # in the value expression
                            val_expr = node.children[1]
                            if len(tensor_node.children) > 1:
                                # Multiple tensors - check if this specific tensor is used
                                modified_expr = self._replace_multi_tensor_loads(val_expr, i)
                                if self._expression_contains_tensor(modified_expr, tensor_name):
                                    if self._is_accumulation_pattern(modified_expr, tensor_name):
                                        accumulators.add(tensor_name)
                                    elif in_sloop and self._expression_contains_tensor(modified_expr, tensor_name):
                                        accumulators.add(tensor_name)
                            else:
                                # Single tensor - original logic
                                if self._expression_contains_tensor(val_expr, tensor_name):
                                    if self._is_accumulation_pattern(val_expr, tensor_name):
                                        accumulators.add(tensor_name)
                                    elif in_sloop and self._expression_contains_tensor(val_expr, tensor_name):
                                        accumulators.add(tensor_name)
        
        def traverse(node: ASTNode, in_sloop: bool = False):
            """Traverse AST to find all store operations"""
            if node.node_type == NodeType.STORE:
                analyze_store(node, in_sloop)
            elif node.node_type == NodeType.SLOOP:
                # Mark that we're inside an sloop for child nodes
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop=True)
            elif node.node_type in [NodeType.SEQ, NodeType.PLOOP]:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop)
        
        traverse(ast)
        return accumulators
    
    def _identify_cross_sloop_tensors(self, ast: ASTNode) -> set:
        """Identify tensors that are used across different sloop scopes"""
        cross_sloop_tensors = set()
        tensor_sloop_map = {}  # tensor_name -> set of sloop IDs where it's used
        sloop_counter = [0]  # Use list to allow modification in nested function
        
        def collect_tensor_usage(node: ASTNode, current_sloop_id: int = -1):
            """Collect which tensors are used in which sloop"""
            if node.node_type == NodeType.SLOOP:
                # New sloop context
                sloop_counter[0] += 1
                new_sloop_id = sloop_counter[0]
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_usage(child, new_sloop_id)
            elif node.node_type == NodeType.STORE:
                # Track tensor being stored
                tensor_node = node.children[0] if node.children else None
                if tensor_node and tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    # Handle multiple tensor names (comma-separated)
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name not in tensor_sloop_map:
                                tensor_sloop_map[tensor_name] = set()
                            if current_sloop_id != -1:  # Only track if inside an sloop
                                tensor_sloop_map[tensor_name].add(current_sloop_id)
                # Also check expression being stored for tensor loads
                if len(node.children) > 1:
                    collect_tensor_usage(node.children[1], current_sloop_id)
                # Continue with other children
                for i in range(2, len(node.children)):
                    if isinstance(node.children[i], ASTNode):
                        collect_tensor_usage(node.children[i], current_sloop_id)
            elif node.node_type == NodeType.LOAD:
                # Track tensor being loaded
                tensor_node = node.children[0] if node.children else None
                if tensor_node and tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    # Handle multiple tensor names (comma-separated)
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name not in tensor_sloop_map:
                                tensor_sloop_map[tensor_name] = set()
                            if current_sloop_id != -1:  # Only track if inside an sloop
                                tensor_sloop_map[tensor_name].add(current_sloop_id)
            else:
                # Recursively process children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_usage(child, current_sloop_id)
        
        collect_tensor_usage(ast)
        
        # Find tensors used in multiple sloops
        for tensor_name, sloop_ids in tensor_sloop_map.items():
            if len(sloop_ids) > 1:
                cross_sloop_tensors.add(tensor_name)
        
        return cross_sloop_tensors

    def _identify_sloop_intermediate_tensors(self, ast: ASTNode) -> set:
        """Identify intermediate tensors that are defined inside sloop"""
        sloop_intermediate_tensors = set()
        
        def traverse(node: ASTNode, in_sloop: bool = False):
            if node.node_type == NodeType.SLOOP:
                # Mark that we're inside an sloop for child nodes
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop=True)
            elif node.node_type == NodeType.STORE and in_sloop:
                # Check if this is storing to an intermediate tensor
                tensor_node = node.children[0]
                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    # Handle multiple tensor names (comma-separated)
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name in self.intermediate_tensors:
                                sloop_intermediate_tensors.add(tensor_name)
                # Continue traversing children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop)
            else:
                # Continue traversing children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop)
        
        traverse(ast)
        return sloop_intermediate_tensors
    
    def _identify_sloop_intermediate_tensors_without_loop_var(self, ast: ASTNode) -> dict:
        """Identify intermediate tensors that are defined inside sloop and don't use loop variable"""
        sloop_tensors_no_loop_var = {}  # {tensor_name: [(sloop_node, loop_var), ...]}
        
        def check_expression_uses_loop_var(expr: ASTNode, loop_var: str) -> bool:
            """Check if an expression uses the loop variable"""
            if expr.node_type == NodeType.VAR and expr.value == loop_var:
                return True
            
            # Check children recursively
            for child in expr.children:
                if isinstance(child, ASTNode):
                    if check_expression_uses_loop_var(child, loop_var):
                        return True
            return False
        
        def traverse(node: ASTNode, in_sloop: bool = False, current_sloop_node: ASTNode = None, current_loop_var: str = None):
            if node.node_type == NodeType.SLOOP:
                # Extract loop variable from sloop node
                # (sloop start end tile_size loop_var body)
                loop_var = node.children[3].value
                # Mark that we're inside an sloop for child nodes
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop=True, current_sloop_node=node, current_loop_var=loop_var)
            elif node.node_type == NodeType.STORE and in_sloop:
                # Check if this is storing to an intermediate tensor
                tensor_node = node.children[0]
                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    # Handle multiple tensor names (comma-separated)
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name in self.intermediate_tensors:
                                # Check if the store expression uses the loop variable
                                uses_loop_var = False
                                if len(node.children) > 1:
                                    # Check the expression being stored
                                    uses_loop_var = check_expression_uses_loop_var(node.children[1], current_loop_var)
                                
                                # Also check index expressions
                                if not uses_loop_var and len(node.children) > 2:
                                    for i in range(2, len(node.children)):
                                        if isinstance(node.children[i], ASTNode):
                                            if check_expression_uses_loop_var(node.children[i], current_loop_var):
                                                uses_loop_var = True
                                                break
                                
                                if not uses_loop_var:
                                    if tensor_name not in sloop_tensors_no_loop_var:
                                        sloop_tensors_no_loop_var[tensor_name] = []
                                    sloop_tensors_no_loop_var[tensor_name].append((current_sloop_node, current_loop_var))
                
                # Continue traversing children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop, current_sloop_node, current_loop_var)
            else:
                # Continue traversing children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop, current_sloop_node, current_loop_var)
        
        traverse(ast)
        return sloop_tensors_no_loop_var
    
    def _expression_uses_var(self, expr: ASTNode, var_name: str) -> bool:
        """Check if an expression uses a specific variable"""
        if expr.node_type == NodeType.VAR and expr.value == var_name:
            return True
        for child in expr.children:
            if isinstance(child, ASTNode) and self._expression_uses_var(child, var_name):
                return True
        return False
    
    def _is_accumulation_in_sloop(self, sloop_node: ASTNode, tensor_name: str) -> bool:
        """Check if a tensor is used in accumulation pattern within a specific sloop"""
        def check_stores_in_node(node: ASTNode) -> bool:
            if node.node_type == NodeType.STORE:
                tensor_node = node.children[0]
                if (tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR] and
                    tensor_node.children[0].value == tensor_name):
                    # Check if this is an accumulation pattern
                    if len(node.children) > 1:
                        return self._is_accumulation_pattern(node.children[1], tensor_name)
            
            # Check children
            for child in node.children:
                if isinstance(child, ASTNode):
                    if check_stores_in_node(child):
                        return True
            return False
        
        # Check the body of the sloop
        if len(sloop_node.children) > 4:
            body = sloop_node.children[4]
            return check_stores_in_node(body)
        return False
    
    def _expression_contains_tensor(self, expr: ASTNode, tensor_name: str) -> bool:
        """Check if an expression contains a load of the given tensor"""
        if expr.node_type == NodeType.LOAD:
            tensor_node = expr.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                # Handle multiple tensor names (comma-separated)
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR and child.value == tensor_name:
                        return True
                return False
        
        # Recursively check children
        for child in expr.children:
            if isinstance(child, ASTNode):
                if self._expression_contains_tensor(child, tensor_name):
                    return True
        return False
    
    def _is_accumulation_pattern(self, expr: ASTNode, tensor_name: str) -> bool:
        """Check if expression is an accumulation pattern (addition involving the tensor)"""
        if expr.node_type == NodeType.ADD:
            # Check if any operand loads the tensor
            for child in expr.children:
                if isinstance(child, ASTNode):
                    if self._expression_contains_tensor(child, tensor_name):
                        return True
        
        # Also check for patterns like (+ (x factor (load tensor)) result)
        for child in expr.children:
            if isinstance(child, ASTNode):
                if self._is_accumulation_pattern(child, tensor_name):
                    return True
        
        return False
    
    def _get_allocated_tensor_shape(self, tensor_name: str) -> list:
        """Get the actual allocated shape for a tensor based on its index pattern"""
        shape = self.tensor_shapes.get(tensor_name, [])
        shape_params = []
        
        # Check if we have index pattern information for this tensor
        if hasattr(self, 'intermediate_tensor_indices') and tensor_name in self.intermediate_tensor_indices:
            index_node = self.intermediate_tensor_indices[tensor_name]
            # Analyze each dimension of the index
            for i, child in enumerate(index_node.children):
                if child.node_type == NodeType.FULLTILE:
                    # For fulltile, use the full dimension from tensor_shapes
                    if i < len(shape):
                        dim = shape[i]
                        shape_params.append(dim)
                    else:
                        shape_params.append(f"{tensor_name}_dim{i}")
                elif child.node_type == NodeType.TILE:
                    # For tile, use the corresponding BLOCK parameter
                    if child.children and child.children[0].node_type == NodeType.VAR:
                        loop_var = child.children[0].value
                        block_param = f"BLOCK_{loop_var.upper()}"
                        shape_params.append(block_param)
                    else:
                        # Fallback to original dimension
                        if i < len(shape):
                            shape_params.append(shape[i])
                elif child.node_type == NodeType.ELEM:
                    # For elem, this dimension is size 1
                    shape_params.append(1)
        else:
            # No index pattern info, use original shape
            shape_params = shape
        
        return shape_params
    
    def _generate_intermediate_allocations(self, ast: ASTNode = None) -> str:
        """Generate tl.zeros allocations for intermediate tensors"""
        # Only allocate intermediate tensors that are NOT cross-kernel tensors
        local_intermediates = self.intermediate_tensors - self.cross_kernel_tensors
        
        # Track temporary variables that will be created for sloop non-accumulation stores
        sloop_temp_vars_shapes = {}
        
        if not local_intermediates and not sloop_temp_vars_shapes:
            return ""
        
        # Identify which tensors are accumulators
        accumulators = set()
        # Identify which tensors are used across different sloops
        cross_sloop_tensors = set()
        # Identify intermediate tensors that are defined inside sloop
        sloop_intermediate_tensors = set()
        if ast:
            accumulators = self._identify_accumulators(ast)
            cross_sloop_tensors = self._identify_cross_sloop_tensors(ast)
            sloop_intermediate_tensors = self._identify_sloop_intermediate_tensors(ast)
            
            # Identify tensors that will need temp vars in sloop
            sloop_tensors_no_loop_var = self._identify_sloop_intermediate_tensors_without_loop_var(ast)
            temp_counter_saved = self.temp_counter
            
            # Check each sloop individually for accumulation patterns
            for tensor_name in sloop_tensors_no_loop_var:
                
                if tensor_name in self.intermediate_tensors:
                    # For each sloop where this tensor appears without loop var
                    for sloop_node, loop_var in sloop_tensors_no_loop_var[tensor_name]:
                        # Check if in THIS specific sloop, the tensor is used in a non-accumulation pattern
                        is_accumulation_in_this_sloop = self._is_accumulation_in_sloop(sloop_node, tensor_name)
                        
                        if not is_accumulation_in_this_sloop:
                            # Generate the same temp var name that will be used later
                            # Use a deterministic naming scheme based on tensor name
                            temp_var_name = f"{tensor_name}_tmp"
                            
                            # Get the actual allocated shape for this tensor
                            allocated_shape = self._get_allocated_tensor_shape(tensor_name)
                            sloop_temp_vars_shapes[temp_var_name] = allocated_shape
                            
                            # Also store mapping for later use
                            if not hasattr(self, 'sloop_tensor_temp_mapping'):
                                self.sloop_tensor_temp_mapping = {}
                            self.sloop_tensor_temp_mapping[tensor_name] = temp_var_name
                            # Only create one temp var per tensor, even if used in multiple sloops
                            break
        
        code = "    # Allocate intermediate tensors\n"
        
        for tensor_name in sorted(local_intermediates):
            if tensor_name in self.tensor_shapes:
                shape = self.tensor_shapes[tensor_name]
                # Build shape string using constants or literal values
                shape_params = []
                
                # Check if we have index pattern information for this tensor
                if hasattr(self, 'intermediate_tensor_indices') and tensor_name in self.intermediate_tensor_indices:
                    index_node = self.intermediate_tensor_indices[tensor_name]
                    # Analyze each dimension of the index
                    for i, child in enumerate(index_node.children):
                        if child.node_type == NodeType.FULLTILE:
                            # For fulltile, use the full dimension from tensor_shapes
                            if i < len(shape):
                                dim = shape[i]
                                if isinstance(dim, str):
                                    shape_params.append(dim)
                                else:
                                    shape_params.append(str(dim))
                            else:
                                shape_params.append(f"{tensor_name}_dim{i}")
                        elif child.node_type == NodeType.TILE:
                            # For tile, use the corresponding BLOCK parameter
                            if child.children and child.children[0].node_type == NodeType.VAR:
                                loop_var = child.children[0].value
                                # Check if this is actually a loop variable in all_loops
                                is_loop_var = any(lv == loop_var for lv, _, _, _ in self.all_loops)
                                if is_loop_var:
                                    block_param = f"BLOCK_{loop_var.upper()}"
                                    shape_params.append(block_param)
                                else:
                                    # Not a loop variable, but still create BLOCK parameter
                                    block_param = f"BLOCK_{loop_var.upper()}"
                                    shape_params.append(block_param)
                            else:
                                # Fallback to original dimension
                                if i < len(shape):
                                    dim = shape[i]
                                    if isinstance(dim, str):
                                        shape_params.append(dim)
                                    else:
                                        shape_params.append(str(dim))
                        elif child.node_type == NodeType.ELEM:
                            # For elem, this dimension is size 1
                            shape_params.append("1")
                else:
                    # No index pattern info, use original logic
                    for dim in shape:
                        if isinstance(dim, str):
                            # It's a symbolic dimension or expression
                            # For simple variables, use them directly in the code
                            # For expressions, resolve them to values
                            if dim in self.constants:
                                # Simple variable, use it directly
                                shape_params.append(dim)
                            elif any(op in dim for op in ['+', '-', '*', '//']):
                                # Expression, resolve it to a value
                                resolved = self.resolve_value(dim)
                                shape_params.append(resolved)
                            elif dim.startswith('BLOCK_'):
                                # Block parameter, use as is
                                shape_params.append(dim)
                            else:
                                # Unknown variable, use as is (might cause error)
                                shape_params.append(dim)
                        else:
                            # It's a literal number
                            shape_params.append(str(dim))
                
                # Ensure single-element tuples have trailing comma
                if len(shape_params) == 1:
                    shape_str = f"({shape_params[0]},)"
                else:
                    shape_str = f"({', '.join(shape_params)})"
                
                # Initialize with zeros if it's an accumulator OR if it's used across different sloops OR if it's defined inside sloop
                if tensor_name in accumulators or tensor_name in cross_sloop_tensors or tensor_name in sloop_intermediate_tensors:
                    code += f"    {tensor_name} = tl.zeros({shape_str}, dtype=tl.float16)\n"
                # else:
                #     # For non-accumulator tensors, just comment that they'll be assigned
                #     code += f"    # {tensor_name} will be assigned without initialization\n"
            # else:
            #     # If shape not provided, default to 2D
            #     if tensor_name in accumulators:
            #         code += f"    {tensor_name} = tl.zeros(({tensor_name}_dim0, {tensor_name}_dim1), dtype=tl.float32)\n"
            #     else:
            #         code += f"    # {tensor_name} will be assigned without initialization\n"
        
        # Also allocate temporary variables for sloop non-accumulation stores
        for temp_var_name in sorted(sloop_temp_vars_shapes):
            shape = sloop_temp_vars_shapes[temp_var_name]
            # Build shape string using constants or literal values
            shape_params = []
            for dim in shape:
                if isinstance(dim, str):
                    shape_params.append(dim)
                else:
                    shape_params.append(str(dim))
            # Ensure single-element tuples have trailing comma
            if len(shape_params) == 1:
                shape_str = f"({shape_params[0]},)"
            else:
                shape_str = f"({', '.join(shape_params)})"
            code += f"    {temp_var_name} = tl.zeros({shape_str}, dtype=tl.float16)\n"
        
        code += "\n"
        return code

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
        # Generate Triton kernel function header with only stride parameters (no shape)
        params = []
        for name in sorted(self.tensors_used):
            # Skip intermediate tensors that are NOT cross-kernel tensors
            if name in self.intermediate_tensors and name not in self.cross_kernel_tensors:
                continue
                
            # pointer
            params.append(f"{name}_ptr")
            
            # Determine number of dimensions based on tensor_shapes or parallel_dims
            num_dims = 2  # default
            if self.tensor_shapes and name in self.tensor_shapes:
                shape = self.tensor_shapes[name]
                # Count the actual dimensions (could be symbolic)
                num_dims = len(shape)
            else:
                # Infer from parallel dimensions
                num_dims = max(len(self.parallel_dims), 1)
            
            # NO shape parameters - will use constants instead
            
            # Stride parameters (still needed as they're runtime values)
            for dim in range(num_dims):
                params.append(f"{name}_stride{dim}: tl.constexpr")
        
        # NO shape parameters for local intermediate tensors either
        
        # block sizes for all loops (parallel and sequential)
        for loop_var, start, end, tile_size in self.all_loops:
            params.append(f"BLOCK_{loop_var.upper()}: tl.constexpr")
            
        # Add constants as parameters
        for const_name in sorted(self.constants.keys()):
            params.append(f"{const_name}: tl.constexpr")
            
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
    
    def _generate_constants_definition(self) -> str:
        """Generate constant definitions at the beginning of the kernel."""
        if not self.constants:
            return ""
            
        code = "    # Constants\n"
        # Sort constants for consistent output
        for const_name in sorted(self.constants.keys()):
            code += f"    {const_name} = {self.constants[const_name]}\n"
        code += "\n"
        return code
    
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
        elif node.node_type == NodeType.SQR:
            return self._generate_binary_op(node, "*")
        elif node.node_type == NodeType.SQRT:
            return self._generate_unary_op(node, "tl.sqrt")
        elif node.node_type == NodeType.SIGMOID:
            return self._generate_unary_op(node, "tl.sigmoid")
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
        elif node.node_type == NodeType.TENSOR:
            return self._generate_tensor(node)
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
        
        code = f"{indent}# Parallel loop {loop_var} from {start} to {end_param} with tile size {block_param}\n"
        code += f"{indent}# Executed across grid dimension {grid_dim}\n"
        code += f"{indent}{loop_var} = {start} + tl.program_id({grid_dim}) * {block_param}\n"
        code += f"{indent}\n"
        # Generate body without boundary check
        body_code = self._generate_node(body)
        # Remove any leading/trailing newlines from body to avoid double newlines
        body_code = body_code.strip('\n')
        if body_code:
            code += body_code + '\n'
        else:
            # Empty body
            code += f"{indent}pass\n"
        
        return code
    
    def _generate_sloop(self, node: ASTNode) -> str:
        """Generate sequential loop code - executed as traditional for loop"""
        # (sloop start end tile_size loop_var body)
        start = self.resolve_value(self._generate_node(node.children[0]))
        end = self.resolve_value(self._generate_node(node.children[1]))
        tile_size = self._generate_node(node.children[2])  # No need to resolve - used as BLOCK parameter
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
        
        # For sloop, use the end value directly (whether it's a number or variable)
        end_param = end
        
        code = f"{indent}# Sequential loop {loop_var} from {start} to {end_param} with tile size {block_param}\n"
        code += f"{indent}for {loop_var} in range({start}, {end_param}, {block_param}):\n"
        
        # Clear load cache for loop body since loop variable changes
        # We need to reload tensors that depend on the loop variable
        saved_load_cache = self.load_cache.copy()
        self.load_cache = {}
        
        # Set current sloop context
        saved_sloop_info = self.current_sloop_info
        self.current_sloop_info = (loop_var, node)
        
        self.indent_level += 1
        
        body_code = self._generate_node(body)
        # Remove any leading/trailing newlines from body to avoid double newlines
        body_code = body_code.strip('\n')
        if body_code:
            code += body_code + '\n'
        else:
            # Empty body
            code += f"{indent}    pass\n"
        self.indent_level -= 1
        
        # Restore load cache after loop
        self.load_cache = saved_load_cache
        
        # Restore previous sloop context
        self.current_sloop_info = saved_sloop_info
        
        return code
    
    def _generate_seq(self, node: ASTNode) -> str:
        """Generate sequence of operations (for nested seq nodes only)"""
        code = ""
        for child in node.children:
            # Skip dummy nodes in sequences
            if child.node_type == NodeType.DUMMY:
                continue
            child_code = self._generate_node(child)
            if child_code:  # Only add if there's actual code
                code += child_code
                # Only add newline if the code doesn't already end with one
                if not child_code.endswith('\n'):
                    code += '\n'
        return code
    
    def _generate_load(self, node: ASTNode) -> str:
        """Generate load operation"""
        # (load tensor index)
        tensor_node = node.children[0]
        index_node = node.children[1]
        
        tensor_name = tensor_node.children[0].value
        
        # Check if we have a temporary variable for this tensor (from sloop non-accumulation store)
        if tensor_name in self.sloop_temp_vars:
            node.temp_var = self.sloop_temp_vars[tensor_name]
            return ""
        
        # Check if this is a local intermediate tensor (not cross-kernel)
        if tensor_name in self.intermediate_tensors and tensor_name not in self.cross_kernel_tensors:
            # For local intermediate tensors, directly reference without reshape
            # tl.dot can handle 3D tensors with batch dimension
            node.temp_var = tensor_name
            return ""
        
        # For input/output tensors and cross-kernel tensors, use normal load operation
        offset_expr = self._generate_index(index_node, tensor_name)
        
        # Create a cache key from tensor name and offset expression
        cache_key = (tensor_name, offset_expr)
        
        # Check if we've already loaded this exact tensor+offset combination
        if cache_key in self.load_cache:
            # Reuse the existing temp variable
            node.temp_var = self.load_cache[cache_key]
            return ""  # No code to generate, just reuse existing variable
        
        # Generate unique variable names for offset
        offset_var = f"offset_{self.offset_counter}"
        self.offset_counter += 1
        
        # Use temp_{counter} for safer variable naming
        var_name = f"temp_{self.temp_counter}"
        self.temp_counter += 1
        
        # Store the variable mapping for later use
        node.temp_var = var_name
        
        # Cache this load operation
        self.load_cache[cache_key] = var_name
        
        # Generate code with separate offset variable
        # Use proper indentation based on current context
        indent_str = '    ' * self.indent_level
        code = f"{indent_str}{offset_var} = {offset_expr}\n"
        code += f"{indent_str}{var_name} = tl.load({tensor_name}_ptr + {offset_var})"
        
        # No need for expand_dims anymore - proper offset calculation handles all dimensions
        # The loaded tensor will have the correct shape based on the offset dimensions
        
        return code
    
    def _replace_multi_tensor_loads(self, node: ASTNode, index: int) -> ASTNode:
        """Replace multi-tensor loads with single tensor at given index"""
        import copy
        
        # Deep copy the node to avoid modifying the original
        new_node = copy.deepcopy(node)
        
        def replace_loads(n):
            if n.node_type == NodeType.LOAD:
                tensor_node = n.children[0]
                # Check if this load has multiple tensor names
                if len(tensor_node.children) > 1 and index < len(tensor_node.children):
                    # Replace with single tensor at index
                    single_child = tensor_node.children[index]
                    n.children[0] = ASTNode(tensor_node.node_type, [single_child])
            
            # Recursively process children
            for child in n.children:
                if isinstance(child, ASTNode):
                    replace_loads(child)
        
        replace_loads(new_node)
        return new_node
    
    def _generate_store(self, node: ASTNode) -> str:
        """Generate store operation"""
        # (store tensor val index)
        tensor_node = node.children[0]
        val_node = node.children[1]
        index_node = node.children[2]
        
        # Check if tensor_node has multiple children (comma-separated tensors)
        if len(tensor_node.children) > 1:
            # Multiple tensors - generate a store for each one
            code = ""
            for i, tensor_child in enumerate(tensor_node.children):
                # Create a new store node for each tensor
                single_tensor_node = ASTNode(tensor_node.node_type, [tensor_child])
                
                # Create a modified value node that uses the corresponding input/tensor
                modified_val_node = self._replace_multi_tensor_loads(val_node, i)
                
                single_store_node = ASTNode(node.node_type, [single_tensor_node, modified_val_node, index_node])
                code += self._generate_store(single_store_node)
                if code and not code.endswith('\n'):
                    code += '\n'
            return code.rstrip('\n')  # Remove trailing newline
        
        tensor_name = tensor_node.children[0].value
        tensor_type = tensor_node.node_type
        
        # First generate any load operations in the value expression
        code = ""
        
        # Check if value is a simple transformation operation (permute, squeeze, unsqueeze)
        # that we can generate inline without temp variable
        if val_node.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # For these operations, generate them inline and assign directly to target tensor
            if val_node.node_type == NodeType.PERMUTE3:
                child = val_node.children[0]
                # Generate child if needed (could be LOAD, UNSQUEEZE, or other operations)
                if not hasattr(child, 'temp_var'):
                    # For UNSQUEEZE nodes, we need special handling to avoid inline generation
                    if child.node_type == NodeType.UNSQUEEZE:
                        # Generate the unsqueeze operation properly
                        unsqueeze_child = child.children[0]
                        if not hasattr(unsqueeze_child, 'temp_var'):
                            child_code = self._generate_node(unsqueeze_child)
                            if child_code:
                                code += child_code
                                if not code.endswith('\n'):
                                    code += '\n'
                        
                        # Now generate the unsqueeze itself
                        child_code = self._generate_node(child)
                        if child_code:
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'
                    else:
                        child_code = self._generate_node(child)
                        if child_code:  # If code was generated, append it
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'
                
                # Get tensor expression - at this point child should have temp_var
                if hasattr(child, 'temp_var'):
                    tensor_expr = child.temp_var
                else:
                    # This should not happen anymore with proper generation
                    raise ValueError(f"Expected temp_var for {child.node_type} node")
                
                # Get permutation dimensions
                dim0 = self._generate_node(val_node.children[1])
                dim1 = self._generate_node(val_node.children[2])
                dim2 = self._generate_node(val_node.children[3])
                
                value_expr = f"tl.permute({tensor_expr}, ({dim0}, {dim1}, {dim2}))"
            
            elif val_node.node_type == NodeType.UNSQUEEZE:
                child = val_node.children[0]
                # Generate child if needed
                if child.node_type == NodeType.LOAD and not hasattr(child, 'temp_var'):
                    code += self._generate_node(child) + "\n"
                
                # Get tensor expression
                tensor_expr = child.temp_var if hasattr(child, 'temp_var') else self._generate_node(child)
                
                # Get dimension
                dim = self._generate_node(val_node.children[1])
                
                value_expr = f"tl.expand_dims({tensor_expr}, {dim})"
            
            elif val_node.node_type == NodeType.SQUEEZE:
                child = val_node.children[0]
                # Generate child if needed
                if child.node_type in [NodeType.LOAD, NodeType.PERMUTE3] and not hasattr(child, 'temp_var'):
                    code += self._generate_node(child)
                    if not code.endswith('\n'):
                        code += '\n'
                
                # Get tensor expression
                tensor_expr = child.temp_var if hasattr(child, 'temp_var') else self._generate_node(child)

                # Get dimension (for squeeze, we use reshape instead)
                # Infer the source tensor name for dimension info
                source_tensor_name = self._infer_tensor_name(child)
                
                # For squeeze operations, we need to use the source tensor's dimensions
                # after removing the squeezed dimension
                if source_tensor_name and source_tensor_name in self.tensor_shapes:
                    # Get the squeezed dimension
                    squeeze_dim = int(self._generate_node(val_node.children[1]))
                    
                    # Check if child was a permute operation to map dimensions correctly
                    if child.node_type == NodeType.PERMUTE3 and hasattr(child, 'permute_dims'):
                        # Get the permuted dimension order
                        perm_dims = child.permute_dims  # e.g., (1, 0, 2)
                        shape_parts = []
                        
                        # Build shape from source tensor dimensions, excluding squeezed dim
                        for i in range(len(perm_dims)):
                            if i != squeeze_dim:  # Skip the dimension to be squeezed
                                orig_dim = perm_dims[i]
                                # Get the dimension value from tensor_shapes
                                shape = self.tensor_shapes[source_tensor_name]
                                if orig_dim < len(shape):
                                    dim_value = shape[orig_dim]
                                    if isinstance(dim_value, str):
                                        # Symbolic dimension
                                        shape_parts.append(dim_value)
                                    else:
                                        # Literal number
                                        shape_parts.append(str(dim_value))
                                else:
                                    # Fallback
                                    shape_parts.append(f"{source_tensor_name}_dim{orig_dim}")
                    else:
                        # No permutation, use original dimension order
                        shape_parts = []
                        num_dims = len(self.tensor_shapes[source_tensor_name])
                        shape = self.tensor_shapes[source_tensor_name]
                        for i in range(num_dims):
                            if i != squeeze_dim:  # Skip the dimension to be squeezed
                                dim_value = shape[i]
                                if isinstance(dim_value, str):
                                    # Symbolic dimension
                                    shape_parts.append(dim_value)
                                else:
                                    # Literal number
                                    shape_parts.append(str(dim_value))
                    
                    shape_str = f"({', '.join(shape_parts)})"
                    value_expr = f"tl.reshape({tensor_expr}, {shape_str})"
                else:
                    # Fallback: use source tensor shape after squeeze
                    # For squeeze operation, we need to use the dimensions of the source tensor O
                    # not the output tensor O2
                    if source_tensor_name and source_tensor_name in self.tensor_shapes:
                        # Get source tensor shape and remove squeezed dimension
                        squeeze_dim = int(self._generate_node(val_node.children[1]))
                        shape_parts = []
                        source_shape = self.tensor_shapes[source_tensor_name]
                        
                        for i in range(len(source_shape)):
                            if i != squeeze_dim:  # Skip the squeezed dimension
                                dim_value = source_shape[i]
                                if isinstance(dim_value, str):
                                    shape_parts.append(dim_value)
                                else:
                                    shape_parts.append(str(dim_value))
                        
                        shape_str = f"({', '.join(shape_parts)})"
                        value_expr = f"tl.reshape({tensor_expr}, {shape_str})"
                    elif tensor_name in self.tensor_shapes:
                        # If we still don't have source tensor info, use output tensor dimensions
                        shape_dims = []
                        for i in range(len(self.tensor_shapes[tensor_name])):
                            dim_value = self.tensor_shapes[tensor_name][i]
                            if isinstance(dim_value, str):
                                # Symbolic dimension - use it directly
                                shape_dims.append(dim_value)
                            else:
                                # Literal number
                                shape_dims.append(str(dim_value))
                        shape_str = f"({', '.join(shape_dims)})"
                        value_expr = f"tl.reshape({tensor_expr}, {shape_str})"
                    else:
                        # Last resort: tensor already reshaped in squeeze operation
                        value_expr = f"{tensor_expr}"

        # Handle other expression types
        elif self._contains_loads(val_node) or self._contains_reduce_sum(val_node):
            code += self._generate_loads_separately(val_node)
            
            # Check if we have nested matmuls and handle them
            if self._contains_nested_matmul(val_node):
                # Find the nested matmul: X @ (A @ B)
                # We need to generate A @ B as a temporary first
                code += self._generate_nested_matmul_temps(val_node)
            
            value_expr = self._generate_node_without_loads(val_node)
        else:
            value_expr = self._generate_node(val_node)
        
        # Use proper indentation based on current context
        indent_str = '    ' * self.indent_level
        
        # Check if this is an output tensor, input tensor, or cross-kernel intermediate tensor
        if tensor_type == NodeType.OUTPUT or tensor_type == NodeType.INPUT or (tensor_type == NodeType.TENSOR and tensor_name in self.cross_kernel_tensors):
            # For output tensors, input tensors, and cross-kernel intermediates, use tl.store
            offset_expr = self._generate_index(index_node, tensor_name)
            
            # Generate unique variable names for offset
            offset_var = f"offset_{self.offset_counter}"
            self.offset_counter += 1
            
            # Generate code with separate offset variable
            code += f"{indent_str}{offset_var} = {offset_expr}\n"
            code += f"{indent_str}tl.store({tensor_name}_ptr + {offset_var}, {value_expr})"
        else:
            # For intermediate tensors (pre-allocated with tl.zeros), use direct assignment
            # Check if index has any tiles or if it's a simple element access
            has_tiles = any(child.node_type in [NodeType.TILE, NodeType.FULLTILE] 
                          for child in index_node.children)
            
            # Check if we're in a sloop and need to create a temporary variable
            should_create_temp = False
            if (self.current_sloop_info and 
                tensor_name in self.intermediate_tensors and
                not self._is_accumulation_pattern(val_node, tensor_name)):
                # Check if the expression uses the loop variable
                loop_var, sloop_node = self.current_sloop_info
                if not self._expression_uses_var(val_node, loop_var):
                    should_create_temp = True
            
            if should_create_temp:
                # Create a temporary variable for this non-accumulation store in sloop
                # Use the pre-determined temp var name
                if hasattr(self, 'sloop_tensor_temp_mapping') and tensor_name in self.sloop_tensor_temp_mapping:
                    temp_var = self.sloop_tensor_temp_mapping[tensor_name]
                else:
                    # Fallback to deterministic naming
                    temp_var = f"{tensor_name}_tmp"
                self.sloop_temp_vars[tensor_name] = temp_var
                code += f"{indent_str}{temp_var} = {value_expr}"
            elif has_tiles:
                # For tiled access, we need to handle accumulation patterns
                # Check if this is an accumulation pattern (common in matrix multiply)
                if val_node.node_type == NodeType.ADD and len(val_node.children) > 0:
                    # Check for (+ (x 1 (load tensor ...)) ...) or (+ (x (load tensor ...) 1) ...)  pattern
                    first_child = val_node.children[0]
                    if (first_child.node_type == NodeType.MUL and 
                        len(first_child.children) == 2):
                        # Check if it's (x 1 load) or (x load 1)
                        is_accumulation = False
                        if (first_child.children[0].node_type == NodeType.NUM and
                            first_child.children[0].value == '1' and
                            first_child.children[1].node_type == NodeType.LOAD):
                            # Pattern: (x 1 load)
                            is_accumulation = True
                        elif (first_child.children[1].node_type == NodeType.NUM and
                              first_child.children[1].value == '1' and
                              first_child.children[0].node_type == NodeType.LOAD):
                            # Pattern: (x load 1)
                            is_accumulation = True
                        
                        if is_accumulation:
                            # This is an accumulation pattern - skip the "* 1" part
                            if len(val_node.children) > 1:
                                new_value = self._generate_node_without_loads(val_node.children[1])
                                code += f"{indent_str}{tensor_name} += {new_value}"
                            else:
                                # Only (x 1 load) with no addition - just assign
                                code += f"{indent_str}{tensor_name} = {value_expr}"
                        else:
                            # Regular addition but not accumulation pattern
                            code += f"{indent_str}{tensor_name} = {value_expr}"
                    else:
                        # Regular addition but not accumulation pattern
                        code += f"{indent_str}{tensor_name} = {value_expr}"
                else:
                    # Direct assignment for non-accumulation patterns
                    code += f"{indent_str}{tensor_name} = {value_expr}"
            else:
                # For element access, direct assignment
                code += f"{indent_str}{tensor_name} = {value_expr}"
        
        return code
    
    def _generate_index(self, node: ASTNode, tensor_name: str) -> str:
        """Generate index calculation for memory access"""
        # (index tile1 tile2 ...)
        offsets = []
        
        for i, child in enumerate(node.children):
            if child.node_type == NodeType.TILE:
                loop_var = child.children[0].value
                if loop_var in self.loop_vars:
                    start, end, tile_size, is_parallel = self.loop_vars[loop_var]
                    
                    # Use BLOCK_SIZE parameter instead of hardcoded tile_size
                    block_param = f"BLOCK_{loop_var.upper()}"
                    
                    # For both parallel and sequential loops, create tile range
                    offset_expr = f"{loop_var} + tl.arange(0, {block_param})"
                    
                    offsets.append(offset_expr)
                else:
                    # Not a loop variable, use a single index (0)
                    offsets.append("0")
            elif child.node_type == NodeType.FULLTILE:
                # For fulltile, we need to determine the appropriate dimension
                # This represents the full dimension being accessed
                # Get the dimension from tensor_shapes
                if tensor_name in self.tensor_shapes:
                    shape = self.tensor_shapes[tensor_name]
                    if i < len(shape):
                        dim_value = shape[i]
                        if isinstance(dim_value, str):
                            # It's a symbolic dimension, use it directly
                            fulltile_dim = dim_value
                        else:
                            # It's a literal number
                            fulltile_dim = str(dim_value)
                    else:
                        # Fallback to parameter name
                        fulltile_dim = f"{tensor_name}_dim{i}"
                else:
                    # Fallback to parameter name
                    fulltile_dim = f"{tensor_name}_dim{i}"
                offsets.append(f"tl.arange(0, {fulltile_dim})")
            elif child.node_type == NodeType.ELEM:
                # For elem indexing, use the loop variable directly without creating a range
                # (elem n) means use n as a scalar index
                if child.children:
                    elem_var = child.children[0].value
                    # Check if this dimension has size 1 in the tensor shape
                    needs_arange = False
                    if tensor_name in self.tensor_shapes:
                        shape = self.tensor_shapes[tensor_name]
                        if i < len(shape):
                            dim_value = shape[i]
                            # Resolve the dimension value to check if it's 1
                            resolved_dim = self.resolve_value(dim_value)
                            if resolved_dim == "1" or resolved_dim == 1:
                                needs_arange = True
                            # Also check for 3D tensors' first dimension (batch/head dimension)
                            elif len(shape) == 3 and i == 0:
                                needs_arange = True
                    
                    if needs_arange:
                        # For dimensions with size 1 or batch dimension in 3D tensors
                        # Use tl.arange(0, 1) to maintain tensor dimensionality
                        block_param = f"BLOCK_{elem_var.upper()}"
                        offsets.append(f"({elem_var}//{block_param})+tl.arange(0, 1)")
                    else:
                        # Divide by tile size to get actual dimension index
                        # elem n with ploop means n can be 0, 64, 128, ... so we need n//BLOCK_N
                        block_param = f"BLOCK_{elem_var.upper()}"
                        offsets.append(f"({elem_var} // {block_param})")
            elif child.node_type == NodeType.CONST_TILE:
                # For const_tile, create a range from start to start+size
                # (const_tile start size) means [start:start+size]
                if len(child.children) >= 2:
                    start = self._generate_node(child.children[0])
                    size = self._generate_node(child.children[1])
                    offsets.append(f"{start} + tl.arange(0, {size})")
        
        # Generate multi-dimensional indexing
        if len(offsets) == 1:
            # For 1D tensors, still need to multiply by stride
            offset_str = f"({offsets[0]}) * {tensor_name}_stride0"
        else:
            # Multi-dimensional indexing: i[:,None]*tensor_stride0 + j[None,:]*tensor_stride1
            offset_parts = []
            
            # Check which dimensions are scalar (elem) vs array (tile/fulltile)
            is_scalar = []
            for off in offsets:
                # Check if offset contains tl.arange (array) or is just a scalar expression
                is_scalar.append("tl.arange" not in off)
            
            for dim, off in enumerate(offsets):
                stride_term = f" * {tensor_name}_stride{dim}"
                
                if is_scalar[dim]:
                    # Scalar index - no broadcasting needed
                    offset_parts.append(f"({off}){stride_term}")
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
                    else:
                        # Need broadcasting
                        broadcast_expr = "[" + ", ".join(nones) + "]"
                        offset_parts.append(f"({off}){broadcast_expr}{stride_term}")
            
            offset_str = " + ".join(offset_parts)
        
        return offset_str
    
    def _generate_binary_op(self, node: ASTNode, op: str) -> str:
        """Generate binary operation"""
        
        if node.node_type == NodeType.SQR:
            # For SQR, generate the operand only once
            child = node.children[0]
            if hasattr(child, 'temp_var'):
                operand = child.temp_var
            else:
                operand = self._generate_node(child)
            return f"({operand} * {operand}).to(tl.float16)"
        
        # For other binary operations
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
            
        return f"({left} {op} {right}).to(tl.float16)"
    
    def _generate_unary_op(self, node: ASTNode, op: str) -> str:
        """Generate unary operation"""
        child = node.children[0]
        
        # Use temp_var if available (from load operations), otherwise generate normally
        if hasattr(child, 'temp_var'):
            operand = child.temp_var
        else:
            operand = self._generate_node(child)
        
        # For exp, sqrt, and sigmoid operations, convert input to float32
        if op in ["tl.exp", "tl.sqrt", "tl.sigmoid"]:
            return f"{op}({operand}.to(tl.float32)).to(tl.float16)"
        else:
            return f"{op}({operand}).to(tl.float16)"
    
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
                    matmul1 = f"tl.dot({left_tensors[0]}, {right_tensors[0]}).to(tl.float16)"
                    matmul2 = f"tl.dot({left_tensors[1]}, {right_tensors[1]}).to(tl.float16)"
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
        return f"tl.dot({left}, {right}).to(tl.float16)"
    
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
            return f"tl.sum({tensor}, axis={axis}, dtype=tl.float16)"
    
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
        # - Along dimension 0: use [None, :] or [None, :, :]
        # - Along dimension 1: use [:, None] or [:, None, :]
        # - Along dimension 2: use [:, :, None]
        if broadcast_dim == 0:
            return f"{tensor_expr}[None, :]"
        elif broadcast_dim == 1:
            return f"{tensor_expr}[:, None]"
        elif broadcast_dim == 2:
            return f"{tensor_expr}[:, :, None]"
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
            
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
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
    
    def _generate_tensor(self, node: ASTNode) -> str:
        """Generate code for intermediate tensor reference"""
        # (tensor C) -> just return the tensor name
        if node.children:
            tensor_name = node.children[0].value
            # Mark this as an intermediate tensor
            self.intermediate_tensors.add(tensor_name)
            return tensor_name
        return "tensor"
    
    def _infer_tensor_shape_from_index(self, index_node: ASTNode) -> tuple:
        """Infer the shape of a tensor from its index pattern"""
        if index_node.node_type != NodeType.INDEX:
            return None
            
        shape = []
        for i, child in enumerate(index_node.children):
            if child.node_type == NodeType.FULLTILE:
                # fulltile represents a full dimension
                # Look for common patterns or use symbolic names
                if i == 0:
                    # First dimension often M (batch/sequence)
                    shape.append('M')
                elif i == 1:
                    # Second dimension could be various things
                    # Check if we're in a context that suggests R (rank)
                    shape.append('R')
                else:
                    shape.append(f'DIM_{i}')
            elif child.node_type == NodeType.TILE:
                # tile with a loop variable
                if child.children:
                    loop_var = child.children[0].value
                    if loop_var in self.loop_vars:
                        _, _, tile_size, _ = self.loop_vars[loop_var]
                        # Use resolved tile size
                        shape.append(self.resolve_value(tile_size))
                    else:
                        # Not a loop variable, use 1 (single tile)
                        shape.append(1)
            elif child.node_type == NodeType.ELEM:
                # elem represents a scalar index
                shape.append(1)
                
        return tuple(shape) if shape else None
    
    def _collect_all_loops(self, node: ASTNode):
        """Collect all loop information for BLOCK parameters"""
        if node.node_type == NodeType.PLOOP:
            # Extract loop info: (ploop start end tile_size loop_var body)
            # Resolve start value (might be a constant variable)
            start_val = node.children[0].value if hasattr(node.children[0], 'value') else "0"
            start = self.resolve_value(start_val)
            
            # Resolve end value (might be a constant variable)
            end_val = node.children[1].value if hasattr(node.children[1], 'value') else "N"
            end = self.resolve_value(end_val)
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
            # Resolve start value (might be a constant variable)
            start_val = node.children[0].value if hasattr(node.children[0], 'value') else "0"
            start = self.resolve_value(start_val)
            
            # Resolve end value (might be a constant variable)
            end_val = node.children[1].value if hasattr(node.children[1], 'value') else "N"
            end = self.resolve_value(end_val)
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
                
                code += f"{indent_str}{temp_name} = tl.dot({left_expr}, {right_expr}).to(tl.float16)\n"
                
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
                        code_line = f"{indent_str}{temp_name} = tl.dot({left_expr}, {right_expr}).to(tl.float16)\n"
                        
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
            load_code = self._generate_node(node)
            if load_code:  # Only add if there's actual code
                code += load_code
                if not load_code.endswith('\n'):
                    code += '\n'
        elif node.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # Generate these operations as well since they produce temp vars
            op_code = self._generate_node(node)
            if op_code:  # Only add if there's actual code
                code += op_code
                if not op_code.endswith('\n'):
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
                return f"tl.dot({left_expr}, {right_expr}).to(tl.float16)"
        elif node.node_type == NodeType.RSUM:
            # Handle reduce sum specially
            child = node.children[0]
            axis = self._generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.sum({child.temp_var}, axis={axis})"
            else:
                child_expr = self._generate_node_without_loads(child)
                return f"tl.sum({child_expr}, axis={axis}, dtype=tl.float16)"
        elif node.node_type == NodeType.BCAST:
            # Handle broadcast specially
            return self._generate_broadcast(node)
        elif node.node_type == NodeType.EXP:
            return self._generate_unary_op(node, "tl.exp")
        elif node.node_type == NodeType.SQR:
            return self._generate_binary_op(node, "*")
        elif node.node_type == NodeType.SQRT:
            return self._generate_unary_op(node, "tl.sqrt")
        elif node.node_type == NodeType.SIGMOID:
            return self._generate_unary_op(node, "tl.sigmoid")
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
        
        # If child doesn't have temp_var yet, generate it
        if not hasattr(child, 'temp_var'):
            if child.node_type in [NodeType.LOAD, NodeType.UNSQUEEZE, NodeType.SQUEEZE, NodeType.PERMUTE3]:
                child_code = self._generate_node(child)
                if child_code and not child_code.endswith('\n'):
                    child_code += '\n'
        
        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            # This should not happen anymore with proper generation
            raise ValueError(f"Expected temp_var for {child.node_type} node in permute3")
        
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
        
        # Store the permutation dimensions for later use (e.g., in squeeze)
        node.permute_dims = (int(dim0), int(dim1), int(dim2))
        
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
        # else:
        #     # For simple expressions like identifiers, evaluate inline
        #     if child.node_type == NodeType.IDENTIFIER:
        #         tensor_expr = child.value
        #     else:
        #         raise ValueError(f"Expected temp_var for {child.node_type} node in squeeze")
        
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
            
            # Check if the child was a permute operation
            if child.node_type == NodeType.PERMUTE3 and hasattr(child, 'permute_dims'):
                # Get the permuted dimension order
                perm_dims = child.permute_dims  # e.g., (1, 0, 2)
                shape_parts = []
                
                # Map the original dimensions through the permutation
                for i in range(num_dims):
                    if i != int(dim):  # Skip the dimension to be squeezed
                        # Find which original dimension this corresponds to
                        orig_dim = perm_dims[i]
                        shape_parts.append(f"{source_tensor_name}_dim{orig_dim}")
            else:
                # No permutation, use original dimension order
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
            elif tensor_node.node_type == NodeType.TENSOR:
                # Handle (tensor name) case
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
        
        # If child doesn't have temp_var yet, generate it
        if not hasattr(child, 'temp_var'):
            if child.node_type in [NodeType.LOAD, NodeType.UNSQUEEZE, NodeType.SQUEEZE, NodeType.PERMUTE3]:
                child_code = self._generate_node(child)
                if child_code and not child_code.endswith('\n'):
                    child_code += '\n'
        
        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        # else:
        #     # For simple expressions like identifiers, evaluate inline
        #     if child.node_type == NodeType.IDENTIFIER:
        #         tensor_expr = child.value
        #     else:
        #         raise ValueError(f"Expected temp_var for {child.node_type} node in unsqueeze")
        
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
