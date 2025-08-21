import re
from typing import List
from NodeType import NodeType
from AstNode import ASTNode

class IRParser:
    """Parser for the tile-level IR AST format"""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize the IR code"""
        # Remove comments and extra whitespace
        code = re.sub(r';.*', '', code)  # Remove comments
        # Split by parentheses, commas, and whitespace, keeping parentheses and commas
        tokens = re.findall(r'\(|\)|,|[^\s(),]+', code)
        return [t for t in tokens if t.strip()]
    
    def parse(self, code: str) -> ASTNode:
        """Parse IR code into AST"""
        self.tokens = self.tokenize(code)
        self.pos = 0
        return self.parse_expression()
    
    def parse_expression(self) -> ASTNode:
        """Parse a single expression"""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of input")
        
        token = self.tokens[self.pos]
        
        if token == '(':
            return self.parse_list()
        elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            self.pos += 1
            return ASTNode(NodeType.NUM, [], int(token))
        else:
            self.pos += 1
            # Special case: treat 'fulltile' as FULLTILE node even without parentheses
            if token == 'fulltile':
                return ASTNode(NodeType.FULLTILE, [])
            # Special case: treat 'dummy' as DUMMY node even without parentheses
            elif token == 'dummy':
                return ASTNode(NodeType.DUMMY, [])
            else:
                return ASTNode(NodeType.VAR, [], token)
    
    def parse_list(self) -> ASTNode:
        """Parse a list expression starting with '('"""
        self.pos += 1  # Skip '('
        
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of input")
        
        op = self.tokens[self.pos]
        self.pos += 1
        
        children = []
        
        # Special handling for tensor, input, output operators with comma-separated names
        if op in ['tensor', 'input', 'output']:
            # Collect all tokens until we hit a non-comma token or ')'
            names = []
            while self.pos < len(self.tokens) and self.tokens[self.pos] != ')':
                if self.tokens[self.pos] == ',':
                    self.pos += 1  # Skip comma
                else:
                    # This should be a name
                    names.append(self.tokens[self.pos])
                    self.pos += 1
                    # Check if next token is comma
                    if self.pos < len(self.tokens) and self.tokens[self.pos] == ',':
                        continue  # Will skip comma in next iteration
                    else:
                        break  # No more names
            
            # If we have multiple names, wrap them in a special node
            if len(names) > 1:
                # Create a list of VAR nodes for the names
                for name in names:
                    children.append(ASTNode(NodeType.VAR, [], name))
            else:
                # Single name, parse as usual
                self.pos -= 1  # Back up one position
                children.append(self.parse_expression())
        else:
            # Normal parsing for other operations
            while self.pos < len(self.tokens) and self.tokens[self.pos] != ')':
                children.append(self.parse_expression())
        
        if self.pos >= len(self.tokens):
            raise ValueError("Missing closing parenthesis")
        
        self.pos += 1  # Skip ')'
        
        # Map operation to node type
        node_type_map = {
            'ploop': NodeType.PLOOP,
            'sloop': NodeType.SLOOP,
            'input': NodeType.INPUT,
            'output': NodeType.OUTPUT,
            'tensor': NodeType.TENSOR,
            'tile': NodeType.TILE,
            'fulltile': NodeType.FULLTILE,
            'index': NodeType.INDEX,
            'load': NodeType.LOAD,
            'store': NodeType.STORE,
            'seq': NodeType.SEQ,
            '+': NodeType.ADD,
            '-': NodeType.SUB,
            'x': NodeType.MUL,
            '/': NodeType.DIV,
            '*': NodeType.MATMUL,
            'exp': NodeType.EXP,
            'sqr': NodeType.SQR,
            'sqrt': NodeType.SQRT,
            'sigmoid': NodeType.SIGMOID,
            'rsum': NodeType.RSUM,
            'concat': NodeType.CONCAT,
            'bcast': NodeType.BCAST,
            'dummy': NodeType.DUMMY,
            'elem': NodeType.ELEM,
            'permute3': NodeType.PERMUTE3,
            'squeeze': NodeType.SQUEEZE,
            'unsqueeze': NodeType.UNSQUEEZE,
            'const_tile': NodeType.CONST_TILE,
        }
        
        if op in node_type_map:
            return ASTNode(node_type_map[op], children)
        else:
            raise ValueError(f"Unknown operation: {op}")
