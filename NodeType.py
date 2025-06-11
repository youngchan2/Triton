from enum import Enum

class NodeType(Enum):
    PLOOP = "ploop"  # Parallel loop
    SLOOP = "sloop"  # Sequential loop
    INPUT = "input"
    TILE = "tile"
    FULLTILE = "fulltile"
    INDEX = "index"
    LOAD = "load"
    STORE = "store"
    SEQ = "seq"
    ADD = "+"
    SUB = "-"
    MUL = "x"
    DIV = "/"
    MATMUL = "*"
    EXP = "exp"
    RSUM = "rsum"
    CONCAT = "concat"
    BCAST = "bcast"
    NUM = "num"
    VAR = "var"