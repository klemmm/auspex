from miasm.expression.expression import *
from miasm.ir.ir import *
import struct
import itertools
import numpy as np
import logging
from typing import Dict, Set, List, Tuple, Optional, Union, Callable, Any
from config import STACK_TOP_ADDR, KSET_MAX, LOG_LEVEL, LOG_FORMAT, TRACKED_DATA_SIZE, MAX_MEM_SIZE
from utils import (
    TOP, BOTTOM, Address, Value, AbstractValue, AbstractAddress,
    AbsRegDict, AbsMemDict, join_val, join_dict, update_dict,
    BITS_TO_NUMPY_TYPE
)
np.seterr(over='ignore', under='ignore')

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

class KSetsDomain(object):
    """Abstract domain for tracking sets of concrete values.
    
    This domain represents abstract values as sets of concrete values,
    with a maximum size limit (KSET_MAX). It tracks both register and
    memory state.
    
    The lattice structure is:
    - TOP (None): represents all possible values
    - Sets: represent specific sets of possible values
    - BOTTOM (empty set): represents no possible values
    """
    
    # ===== Static utility functions =====
    
    @staticmethod
    def _get_bit_size_and_sign_mask(concrete_val: np.unsignedinteger) -> Tuple[int, np.unsignedinteger]:
        """Get the bit size and sign mask for a numpy integer type."""
        bit_size = concrete_val.itemsize * 8
        sign_mask = np.uint64(1) << np.uint64(bit_size - 1) 
        return bit_size, type(concrete_val)(sign_mask)

    @staticmethod
    def _handle_rol(concrete_args: List[np.unsignedinteger]) -> np.unsignedinteger:
        """Handle rotate left operation."""
        concrete_val, amount = concrete_args[0], int(concrete_args[1]) 
        bit_size = concrete_val.itemsize * 8
        roll = amount % bit_size
        if roll == 0: return concrete_val
        val_u64 = np.uint64(concrete_val)
        mask = (np.uint64(1) << np.uint64(bit_size)) - np.uint64(1)
        rotated = ((val_u64 << np.uint64(roll)) | (val_u64 >> np.uint64(bit_size - roll))) & mask
        return type(concrete_val)(rotated)

    @staticmethod
    def _handle_ror(concrete_args: List[np.unsignedinteger]) -> np.unsignedinteger:
        """Handle rotate right operation."""
        concrete_val, amount = concrete_args[0], int(concrete_args[1])
        bit_size = concrete_val.itemsize * 8
        roll = amount % bit_size
        if roll == 0: return concrete_val
        val_u64 = np.uint64(concrete_val)
        mask = (np.uint64(1) << np.uint64(bit_size)) - np.uint64(1)
        rotated = ((val_u64 >> np.uint64(roll)) | (val_u64 << np.uint64(bit_size - roll))) & mask
        return type(concrete_val)(rotated)

    @staticmethod
    def _handle_flag_sign_sub(args: List[np.unsignedinteger]) -> bool:
        """Handle sign flag for subtraction.
        
        Args:
            args: List containing [a, b]
            
        Returns:
            True if result is negative
        """
        a, b = args[0], args[1]
        res = a - b
        bit_size, sign_mask = KSetsDomain._get_bit_size_and_sign_mask(a)
        return bool(res & sign_mask)

    @staticmethod
    def _handle_flag_add_cf(args: List[np.unsignedinteger]) -> bool:
        """Handle carry flag for addition.
        
        Args:
            args: List containing [a, b]
            
        Returns:
            True if addition carries
        """
        a, b = args[0], args[1]
        res = a + b
        return res < a

    @staticmethod
    def _handle_flag_add_of(args: List[np.unsignedinteger]) -> bool:
        """Handle overflow flag for addition.
        
        Args:
            args: List containing [a, b]
            
        Returns:
            True if addition overflows
        """
        a, b = args[0], args[1]
        res = a + b
        bit_size, sign_mask = KSetsDomain._get_bit_size_and_sign_mask(a)
        return bool(((a ^ b) & sign_mask == 0) and ((res ^ a) & sign_mask != 0))

    @staticmethod
    def _handle_flag_sub_cf(args: List[np.unsignedinteger]) -> bool:
        """Handle carry flag for subtraction.
        
        Args:
            args: List containing [a, b]
            
        Returns:
            True if subtraction borrows
        """
        a, b = args[0], args[1]
        return b > a

    @staticmethod
    def _handle_flag_sub_of(args: List[np.unsignedinteger]) -> bool:
        """Handle overflow flag for subtraction.
        
        Args:
            args: List containing [a, b]
            
        Returns:
            True if subtraction overflows
        """
        a, b = args[0], args[1]
        res = a - b
        bit_size, sign_mask = KSetsDomain._get_bit_size_and_sign_mask(a)
        return bool(((a ^ b) & sign_mask != 0) and ((res ^ a) & sign_mask != 0))

    @staticmethod
    def _get_op_handlers() -> Dict[str, Callable]:
        """Get the operation handlers for this domain."""
        return {
                '+': lambda concrete_args: concrete_args[0] + concrete_args[1],
                '-': lambda concrete_args: concrete_args[0] - concrete_args[1] if len(concrete_args) == 2 else -concrete_args[0],
                '&': lambda concrete_args: concrete_args[0] & concrete_args[1],
                '|': lambda concrete_args: concrete_args[0] | concrete_args[1],
                '^': lambda concrete_args: concrete_args[0] ^ concrete_args[1],

                '<<<': KSetsDomain._handle_rol, 
                '>>>': KSetsDomain._handle_ror,

                'CC_EQ': lambda concrete_args: concrete_args[0],   
                'CC_U>': lambda concrete_args: not(concrete_args[0] or concrete_args[1]), 
                'CC_S<': lambda concrete_args: concrete_args[0] != concrete_args[1], 
                'CC_U<=': lambda concrete_args: concrete_args[0] or concrete_args[1],
                'CC_U<': lambda concrete_args: concrete_args[0],

                'FLAG_EQ_CMP': lambda concrete_args: concrete_args[0] == concrete_args[1],

                'FLAG_SIGN_SUB': KSetsDomain._handle_flag_sign_sub, 
                'FLAG_ADD_CF': KSetsDomain._handle_flag_add_cf, 
                'FLAG_ADD_OF': KSetsDomain._handle_flag_add_of,  
                'FLAG_SUB_CF': KSetsDomain._handle_flag_sub_cf,   
                'FLAG_SUB_OF': KSetsDomain._handle_flag_sub_of,  

                'parity': lambda concrete_args: bin(concrete_args[0]).count('1') % 2 == 1, 
                'zeroExt_64': lambda concrete_args: np.uint64(concrete_args[0]),
        }   

    # ===== Private helper functions =====
    
    @staticmethod
    def _concretize(abs_val: AbstractValue) -> Set[Value]:
        """Convert an abstract value to a concrete set."""
        assert(abs_val is not TOP)
        return abs_val

    @staticmethod
    def _abstract(concrete_val: Value) -> Set[Value]:
        """Convert a concrete value to an abstract set."""
        return {concrete_val}
    
    @staticmethod
    def _includes(abs_val: AbstractValue, concrete_val: Value) -> bool:
        """Check if a concrete value is included in an abstract value."""
        return True if abs_val is TOP else concrete_val in abs_val

    # ===== Lattice operations =====
    
    def join(self, other: 'KSetsDomain') -> None:
        """Join this domain with another domain."""
        if other.is_bot:
            return
        if self.is_bot:
            self.assign(other)

        join_dict(self.reg, other.reg)
        join_dict(self.mem, other.mem)

    def equals(self, other: 'KSetsDomain') -> bool:
        """Check if this domain equals another domain."""
        if self.is_bot and other.is_bot:
            return True        
        return self.is_bot == other.is_bot and self.reg == other.reg and self.mem == other.mem

    # ===== State management =====
    
    def assign(self, other: 'KSetsDomain') -> None:
        """Assign the state of another domain to this one."""
        self.is_bot = other.is_bot
        self.reg = other.reg.copy()
        self.mem = other.mem.copy()  

    def set_to_bottom(self) -> None:
        """Set this domain to the bottom element."""
        self.is_bot = True
        self.reg = {}
        self.mem = {}

    def clone(self) -> 'KSetsDomain':
        """Create a copy of this domain."""
        new = KSetsDomain(True)
        new.assign(self)
        return new

    def weak_update(self, concrete_addr: Address, val: AbstractValue, size: int) -> None:
        """Perform a weak update of memory."""

        max_write_size = max(size, TRACKED_DATA_SIZE)                      
        #Conservatively delete all memory addresses that could be affected by the write
        #This is imprecise, but sound. Sufficient for now.
        for a in range(1, max_write_size):
            if concrete_addr + a in self.mem:
                del self.mem[concrete_addr + a]
            if concrete_addr - a in self.mem:
                del self.mem[concrete_addr - a]
        
        if concrete_addr in self.mem:
            if val is TOP or size != TRACKED_DATA_SIZE:
                del self.mem[concrete_addr]
            else:
                self.mem[concrete_addr].update(val)

    def strong_update(self, concrete_addr: Address, abs_val: AbstractValue, size: int) -> None:
        """Perform a strong update of memory."""

        if abs_val is TOP:
            max_write_size = MAX_MEM_SIZE
        else:
            max_write_size = size

        self.weak_update(concrete_addr, TOP, max_write_size) 
        if size == TRACKED_DATA_SIZE:
            logging.debug("Strong update at address %s", hex(concrete_addr))
            update_dict(self.mem, concrete_addr, abs_val)

    def read(self, concrete_addr: Address, nbytes: int) -> AbstractValue:
        """Read a value from memory."""
        if nbytes == 4:
            return self.mem.get(concrete_addr, TOP)
        else:
            return TOP

    # ===== Expression evaluation =====
    
    def lift(self, f: Callable, concrete_args: List[Any], 
             abs_args: List[Any], 
             f_res_is_abstract: bool = False, evaluate_args: bool = True) -> AbstractValue:
        """Lift a concrete function to work on abstract values."""
        evaluated_args = []
        for abs_arg in abs_args:
            if evaluate_args:
                eval_arg = self._evaluate(abs_arg)
            else:
                eval_arg = abs_arg
            if eval_arg is TOP:
                return TOP
            evaluated_args.append(KSetsDomain._concretize(eval_arg))

        results = BOTTOM
        for args in [list(t) for t in itertools.product(*evaluated_args)]:
            r = f(concrete_args, args)
            if not f_res_is_abstract:
                r = KSetsDomain._abstract(r)
            results = join_val(results, r)
        return results

    def _evaluate(self, expr: Expr) -> AbstractValue:
        """Evaluate an expression in this domain."""
        if isinstance(expr, ExprId):
            return self.reg.get(expr, TOP)
        elif isinstance(expr, ExprOp):
            def handle_op(op: str, concrete_args: List[Value]) -> AbstractValue:
                handler = self.op_handlers.get(op)
                if handler:
                    return handler(concrete_args)
                else:
                    raise ValueError(f"Unhandled expr operator: {op}")
            return self.lift(handle_op, expr.op, expr.args)
        elif isinstance(expr, ExprMem):
            handle_mem = lambda size, concrete_addr: self.read(concrete_addr[0], size)
            return self.lift(handle_mem, expr.size // 8, [expr.ptr], True)
        elif isinstance(expr, ExprSlice):
            def handle_slice(slice: Tuple[int, int], sliced: Set[Value]) -> AbstractValue:
                start, stop = slice
                sliced = sliced[0]
                sliced = sliced >> start
                sliced &= (1 << (stop - start)) - 1
                if stop == start + 1:            
                    return [False, True][sliced]            
                elif (stop - start) in BITS_TO_NUMPY_TYPE:
                    return BITS_TO_NUMPY_TYPE[stop - start](sliced)                            
                return (int(sliced), stop - start)                
            
            return self.lift(handle_slice, (expr.start, expr.stop), [expr.arg])
        elif isinstance(expr, ExprInt):                            
            return KSetsDomain._abstract(BITS_TO_NUMPY_TYPE[expr.size](expr.arg))
        elif isinstance(expr, ExprCompose):
            def handle_compose(_, to_compose: List[Set[Value]]) -> AbstractValue:   
                total_size = 0
                result = 0                
                for arg in to_compose:                
                    if type(arg) is tuple:
                        arg_size = arg[1]
                        result |= arg[0] << total_size
                    else:
                        arg_size = arg.itemsize << 3
                        result |= int(arg) << total_size
                    
                    total_size += arg_size
                result = BITS_TO_NUMPY_TYPE[total_size](result)
                return result
            return self.lift(handle_compose, None, expr.args)  
        elif isinstance(expr, ExprLoc):
            return KSetsDomain._abstract(expr.loc_key)
        elif isinstance(expr, ExprCond):
            evaluated_cond = self._evaluate(expr.cond)           
            if KSetsDomain._includes(evaluated_cond, True) and KSetsDomain._includes(evaluated_cond, False):                
                return join_val(self._evaluate(expr.src1), self._evaluate(expr.src2))
            elif KSetsDomain._includes(evaluated_cond, True):
                return self._evaluate(expr.src1)
            elif KSetsDomain._includes(evaluated_cond, False):                
                return self._evaluate(expr.src2)    
            else:
                return BOTTOM                    
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    # ===== Public interface =====
    
    def __init__(self, is_bot: bool) -> None:
        """Initialize the abstract domain."""
        self.is_bot = is_bot
        self.reg: AbsRegDict = {}
        self.mem: AbsMemDict = {}
        self.op_handlers = KSetsDomain._get_op_handlers()
        if not is_bot:
            self.reg[ExprId("RSP", 64)] = {np.uint64(STACK_TOP_ADDR)}
            self.reg[ExprId("RBP", 64)] = {np.uint64(STACK_TOP_ADDR)}

    def incompatible_loc(self, concrete_loc: Optional[LocKey]) -> bool:      
        """Check if a location is incompatible with this domain."""
        return ExprId("IRDst", 64) in self.reg and concrete_loc is not None and concrete_loc not in KSetsDomain._concretize(self.reg[ExprId("IRDst", 64)]) 
               
    def update(self, ir: IRBlock) -> Set[Address]:
        """
        Update this domain with an IR statement.
        
        Returns a set of strongly-updated addresses for dispatch state variable detection heuristic
        """
        if ir.items() == []:
            return set()                 

        logging.debug("Current abstract state: %s", self)
        update_dict(self.reg, ExprId("IRDst", 64), TOP) #don't care about IRDst except after a jump
        logging.debug("Instruction: %s", ir.instr)
        if "CALL" in str(ir.instr):
            logging.warning("CALL not implemented yet")            
            return set()
        logging.debug("IR statements:")
        for dst, src in ir.items():
            logging.debug("  %s := %s", dst, src)

        #/!\ Assignations are done in parallel: First compute all prerequisites
        abs_vals = []
        abs_write_addrs = []
        for dst, src in ir.items():            
            abs_vals.append(self._evaluate(src))
            if isinstance(dst, ExprMem):
                abs_write_addrs.append(self._evaluate(dst.ptr))
            else:
                abs_write_addrs.append(TOP)
        
        #Then, update state
        i = 0
        strongly_updated_addrs = set()
        for dst, _ in ir.items():
            if isinstance(dst, ExprId): 
                update_dict(self.reg, dst, abs_vals[i])  
            elif isinstance(dst, ExprMem):                
                if abs_write_addrs[i] is TOP:
                    # We should clear whole memory dict here to be sound, we don't do it to preserve state value
                    logging.warning("Unknown write address - result may be unsound")                    
                elif abs_vals[i] is TOP:
                    logging.debug("Unknown value - clearing memory at address %s", abs_write_addrs[i])
                    for concrete_addr in KSetsDomain._concretize(abs_write_addrs[i]):
                        self.weak_update(concrete_addr, TOP, dst.size >> 3)
                else:
                    concrete_addrs = KSetsDomain._concretize(abs_write_addrs[i])
                    if len(concrete_addrs) == 1:
                        concrete_addr = next(iter(concrete_addrs))
                        strongly_updated_addrs.add(concrete_addr)
                        self.strong_update(concrete_addr, abs_vals[i], dst.size >> 3)
                    else:
                        for concrete_addr in concrete_addrs:
                            self.weak_update(concrete_addr, abs_vals[i], dst.size >> 3)

            else:
                logging.error("Unhandled assignment LHS type: %s", type(dst))
                raise ValueError("Unhandled assignment LHS type: " + str(type(dst)))
            i = i + 1

        logging.debug("After-interpretation abstract state: %s", self)

        #normalize bottom state
        for k in self.reg:
            if self.reg[k] is BOTTOM:
                logging.debug("Register %s is BOTTOM - setting domain to bottom state", k)
                self.is_bot = True
                return set()
        for k in self.mem:
            if self.mem[k] is BOTTOM:
                logging.debug("Memory at %s is BOTTOM - setting domain to bottom state", hex(k))
                self.is_bot = True
                return set() 
        return strongly_updated_addrs        

    def __str__(self) -> str:
        """Get a string representation of this domain."""
        def showval(concrete_val: Union[Value, bool, int]) -> str:
            if isinstance(concrete_val, (np.uint8, np.uint16, np.uint32, np.uint64)):
                return hex(concrete_val)
            else:
                return str(concrete_val)
        def showset(abs_val: AbstractValue) -> str:
            if abs_val is TOP:
                return "TOP"
            elif abs_val is BOTTOM:
                return "BOTTOM"
            elif type(abs_val) is set:
                return str([showval(v) for v in abs_val])
            else:
                return showval(abs_val)
        if self.is_bot:
            return "BOTTOM"
        else:            
            return "\nreg=" + str({k:showset(v) for k,v in self.reg.items()}) + ",\nmem=" + str({hex(k):showset(v) for k,v in self.mem.items()})            
