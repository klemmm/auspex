from typing import Dict, Set, List, Tuple, Optional, Union, Callable
from miasm.expression.expression import ExprId
import numpy as np
import struct
from config import KSET_MAX

# Type aliases for better readability
TOP = None  # Represents all possible values
BOTTOM = set()  # Represents no possible values
Address = np.uint64  # Type for memory addresses
Value = Union[np.uint8, np.uint16, np.uint32, np.uint64, bool, Tuple[int, int]]  # Type for concrete values
AbstractValue = Union[Set[Value], None]  # None represents TOP
AbstractAddress = Union[Address, None]  # None represents TOP
AbsRegDict = Dict[ExprId, AbstractValue]  # Type for abstract register state
AbsMemDict = Dict[Address, AbstractValue]  # Type for abstract memory state

def join_val(abs_val1: AbstractValue, abs_val2: AbstractValue) -> AbstractValue:
    """Join two abstract values in the lattice.
    
    Args:
        abs_val1: First abstract value
        abs_val2: Second abstract value
        
    Returns:
        The join of the two values in the lattice
    """
    if abs_val1 is TOP or abs_val2 is TOP:
        return TOP
    elif abs_val1 is BOTTOM:
        return abs_val2
    elif abs_val2 is BOTTOM:
        return abs_val1
    else:
        joined = abs_val1.union(abs_val2)
        return TOP if len(joined) > KSET_MAX else joined

def join_dict(abs_dict1: Union[AbsRegDict, AbsMemDict], abs_dict2: Union[AbsRegDict, AbsMemDict]) -> None:
    """Join two dictionaries of abstract values.
    
    Args:
        abs_dict1: First dictionary (modified in place)
        abs_dict2: Second dictionary
    """
    to_remove = []
    #Note: since the dict represents absent keys as implicit top, deleting entries actually represents an increase
    #Therefore, even if it feels like doing an intersection, from the abstract domain point of view it is really an union
    for k, v in abs_dict1.items():
        if k in abs_dict2:
            joined_val = join_val(abs_dict2[k], v)
            if joined_val is TOP:
                to_remove.append(k)
            else:
                abs_dict1[k] = joined_val
        else:
            to_remove.append(k)
    for k in to_remove:
            del abs_dict1[k]

def update_dict(abs_dict: Union[AbsRegDict, AbsMemDict], key: Union[Address, ExprId], abs_val: AbstractValue) -> None:
    """Update a dictionary with a key-value pair.
    
    Args:
        abs_dict: Dictionary to update
        abskey: Key to update
        abs_val: New value
    """
    if abs_val is TOP:
        if key in abs_dict:
            del abs_dict[key]
    else:
        abs_dict[key] = abs_val

# Type conversion mappings
NUMPY_TO_STRUCT_FMT = {
    np.uint8: 'B',
    np.uint16: 'H',
    np.uint32: 'I',
    np.uint64: 'Q'
}

BITS_TO_NUMPY_TYPE = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
} 