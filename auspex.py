#!/usr/bin/env python3
import sys
import logging
from typing import Dict, Set, List, Tuple, Optional, Union, Callable
import domain
from miasm.ir.ir import AssignBlock
from miasm.expression.expression import LocKey
from miasm.analysis.machine import Machine
from miasm.analysis.binary import Container
from miasm.core.locationdb import LocationDB
import traceback

from config import LOG_LEVEL, LOG_FORMAT, TRACKED_DATA_SIZE
import numpy as np
from utils import TOP, Address, Value, AbstractValue, AbstractAddress

# Type aliases for better readability
DispatchState = Optional[Value]  # None represents TOP
StateKey = Tuple[LocKey, DispatchState]
StateTransition = Tuple[AbstractValue, AbstractValue]
Graph = Dict[str, Set[str]]  # Maps state hex strings to sets of successor state hex strings

class Interpreter:
    """Binary analysis interpreter that tracks dispatch state transitions.
    
    This interpreter analyzes a binary by tracking how dispatch states change
    through the program's execution. It builds a state transition graph showing
    how different dispatch states are connected.
    """
    
    def __init__(self, binary_path: str, function_name: str) -> None:
        """Initialize the interpreter with a binary file.
        
        Args:
            binary_path: Path to the binary file to analyze
        """
        self._load_binary(binary_path, function_name)

        self.domain_class = domain.KSetsDomain

    def prepare(self, state_split: bool) -> None:
        # Dispatch state management
        self.state_split = state_split
        self.seen_write_location = set()
        self.address_score = {}
        self.instr_state = {}
        self.instr_disasm = {}  # Store disassembly for each instruction

        # Initialize analysis state
        self.abstract_states: Dict[StateKey, domain.KSetsDomain] = {}
        self.worklist: List[Tuple[LocKey, DispatchState]] = [(self.entry_block, TOP)]
        self.abstract_states[(self.entry_block.key, TOP)] = self.domain_class(False)
        
        # Initialize graph structures
        self.state_graph: Graph = {}
        self.terminal_states: Set[str] = set()        

    def _load_binary(self, binary_path: str, function_name: str) -> None:
        """Load and prepare the binary for analysis.
        
        This sets up the disassembly engine and creates the initial IR blocks.
        
        Args:
            binary_path: Path to the binary file to load
        """
        self.loc_db = LocationDB()
        container = Container.from_stream(open(binary_path, "rb"), self.loc_db)
        machine = Machine(container.arch)

        disasm_engine = machine.dis_engine(container.bin_stream, loc_db=self.loc_db)
        entry_loc = self.loc_db.get_name_location(function_name)
        entry_offset = self.loc_db.get_location_offset(entry_loc)
        if entry_loc is None:
            raise ValueError(f"Function {function_name} not found in binary")        
        asm_cfg = disasm_engine.dis_multiblock(entry_offset)
        lifter = machine.lifter_model_call(self.loc_db)
        self.ir_cfg = lifter.new_ircfg_from_asmcfg(asm_cfg)
        self.entry_block = self.ir_cfg.get_block(entry_loc).loc_key

    def _update_instr_state(self, assignment: AssignBlock, input_state: AbstractValue):
        """Update the instruction-to-state mapping"""
        if self.state_split and input_state is not TOP and len(input_state) == 1:
            if assignment.instr.offset not in self.instr_state:
                self.instr_state[assignment.instr.offset] = set()
                # Store the disassembly for this instruction
                self.instr_disasm[assignment.instr.offset] = str(assignment.instr)
            self.instr_state[assignment.instr.offset].add(next(iter(input_state)))


    def _process_block(self, current_block: LocKey, abs_state: domain.KSetsDomain) -> StateTransition:
        """Process a single block and track dispatch state changes.
        
        Args:
            current_block: The current IR block being processed
            abs_state: The current abstract state
            
        Returns:
            Tuple of (input dispatch states, output dispatch states)
        """
        ir_block = self.ir_cfg.get_block(current_block)
        block_offset = self.loc_db.get_location_offset(current_block)
        
        logging.debug("\nProcessing block at %s (address: %s)", 
                     current_block, hex(block_offset) if block_offset is not None else "<synthetic block>")
        
        # Read input dispatch state
        input_state: AbstractValue = abs_state.read(self.dispatch_state_addr, TRACKED_DATA_SIZE) if self.state_split else TOP 

        logging.debug("Input dispatch state: %s", 
                     str([hex(val) for val in input_state]) if input_state is not TOP else "TOP")
            
        # Process all assignments in the block        
        for assignment in ir_block.assignblks:
            self._update_instr_state(assignment, input_state)

            strongly_updated_addrs = abs_state.update(assignment)
            for addr in strongly_updated_addrs:
                if (current_block.key, addr) not in self.seen_write_location:
                    self.seen_write_location.add((current_block.key, addr))                    
                    self.address_score[addr] = self.address_score.get(addr, 0) + 1
                 
        # Read output dispatch state
        output_state: AbstractValue = abs_state.read(self.dispatch_state_addr, TRACKED_DATA_SIZE) if self.state_split else TOP
        logging.debug("Output dispatch state: %s", 
                     str([hex(val) for val in output_state]) if output_state is not TOP else "TOP")
        
        # If this is a terminal block, mark its states as terminal
        if not list(self.ir_cfg.successors_iter(current_block)) and output_state is not TOP:
            self.terminal_states.update(hex(val) for val in output_state)
            
        return input_state, output_state

    def _update_state_graph(self, input_state: AbstractValue, output_state: AbstractValue) -> None:
        """Update the state transition graph with new edges.
        
        Args:
            input_state: Input dispatch state
            output_state: Output dispatch state
        """
        # Handle transitions from initial state
        if input_state is TOP:
            # TOP represents the initial state
            for target_state in output_state:
                self.state_graph.setdefault("START", set()).add(hex(target_state))
        else:       
            if output_state is TOP:                
                logging.error("Dispatcher state goes to TOP, increase KSET_MAX")
                raise ValueError("Dispatcher state goes to TOP, increase KSET_MAX")
            # Add transitions between states
            for source_state in input_state:
                for target_state in output_state:
                    if target_state != source_state:  # Avoid self-loops
                        self.state_graph.setdefault(hex(source_state), set()).add(hex(target_state))

    def _process_successors(self, current_block: LocKey, abs_state: domain.KSetsDomain, output_state: AbstractValue) -> None:
        """Process all successor blocks of the current location.
        
        Args:
            current_block: Current block being processed
            abs_state: Current abstract state
            output_state: Output dispatch state from current block
        """
        successor_count: int = 0
        for succ_block in self.ir_cfg.successors_iter(current_block):
            succ_offset = self.loc_db.get_location_offset(succ_block)
            succ_addr_str = hex(succ_offset) if succ_offset is not None else "<synthetic block>"
            logging.debug("Exploring edge: %s -> %s (address: %s)", 
                         current_block, succ_block, succ_addr_str)
            
            if abs_state.incompatible_loc(succ_block):
                logging.debug("Cannot transition to %s (impossible branch)", succ_block)
                continue
                
            # Process each possible dispatch state
            possible_states: List[DispatchState] = [None] if output_state is TOP else output_state
            for next_state in possible_states:
                self._process_successor_state(succ_block, abs_state, next_state, succ_addr_str)
                successor_count += 1
                
        logging.debug("Processed %d successor states", successor_count)

    def _process_successor_state(self, succ_block: LocKey, abs_state: domain.KSetsDomain, 
                               next_state: DispatchState, succ_addr_str: str) -> None:
        """Process a single successor state and update the worklist.
        
        Args:
            succ_block: Successor block
            abs_state: Current abstract state
            next_state: Dispatch state to propagate
            succ_addr_str: String representation of the successor block's address
        """
        # Create new state for successor        
        succ_state = abs_state.clone()
        if next_state is not None:
            # Perform Filtering (refinement) of the abstract state based on the known dispatcher state
            succ_state.strong_update(np.uint64(self.dispatch_state_addr), {next_state}, TRACKED_DATA_SIZE)
        
        # Check if we need to update the successor's state
        state_key = (succ_block.key, next_state)
        if state_key in self.abstract_states:
            succ_state.join(self.abstract_states[state_key])
            
        if state_key not in self.abstract_states or not succ_state.equals(self.abstract_states[state_key]):
            self.abstract_states[state_key] = succ_state
            self.worklist.append((succ_block, next_state))
            logging.debug("Scheduling successor %s with state %s", 
                         succ_addr_str, hex(next_state) if next_state is not None else "TOP")
        else:
            logging.debug("Skipping already processed state %s at %s", 
                         hex(next_state) if next_state is not None else "TOP", succ_addr_str)

    def run(self) -> None:
        """Run the analysis until all reachable states are processed."""
        logging.info("Starting analysis...")
        steps = 0
        while self.worklist:
            steps += 1
            current_block, current_state = self.worklist.pop()
            abs_state = self.abstract_states.get((current_block.key, current_state), self.domain_class(True)).clone()
            
            input_state, output_state = self._process_block(current_block, abs_state)
            
            if output_state != input_state:
                self._update_state_graph(input_state, output_state)
                
            if not abs_state.is_bot:
                self._process_successors(current_block, abs_state, output_state)
        logging.info("Abstract interpretation converged in " + str(steps) + " steps.")

    def write_output(self, output_path: str) -> None:
        """Write the state transition graph to a DOT file.
        
        Args:
            output_path: Path to write the DOT file
        """
        logging.info("Writing state transition graph to %s", output_path)
        with open(output_path, "w") as f:
            f.write("digraph finite_state_machine {\n")
            f.write("    rankdir=LR;\n")
            f.write("    node [shape=none, fontname=\"Courier\"];\n")
            f.write("    edge [fontname=\"Courier\"];\n")

            # First, create a mapping of states to their unique instructions
            state_to_instrs = {}
            for instr_offset, states in self.instr_state.items():
                if len(states) == 1:  # Only include instructions that map to exactly one state
                    state = next(iter(states))
                    state_to_instrs.setdefault(hex(state), []).append(instr_offset)
            
            # Sort instructions by address for each state
            for state in state_to_instrs:
                state_to_instrs[state].sort()

            # Find the maximum line width
            max_width = 0
            for state in state_to_instrs:
                for addr in state_to_instrs[state]:
                    line = f"0x{hex(addr)[2:].zfill(8)}: {self.instr_disasm[addr]}"
                    max_width = max(max_width, len(line))

            # Write edges
            for source in self.state_graph:
                for target in self.state_graph[source]:
                    f.write(f'    "{source}" -> "{target}";\n')

            def create_html_label(state, instrs):
                html = ['<']
                html.append('<TABLE CELLBORDER="0" CELLSPACING="0">')
                html.append(f'<TR><TD COLSPAN="2" ALIGN="CENTER"><B>{state}</B></TD></TR>')
                for addr in instrs:
                    addr_str = f"0x{hex(addr)[2:].zfill(8)}"
                    disasm = self.instr_disasm[addr]
                    html.append(f'<TR><TD ALIGN="RIGHT">{addr_str}:</TD><TD ALIGN="LEFT">{disasm}</TD></TR>')
                html.append('</TABLE>')
                html.append('>')
                return ''.join(html)

            # Write nodes with instruction information
            for state in self.state_graph:
                instrs = state_to_instrs.get(state, [])
                label = create_html_label(state, instrs)
                f.write(f'    "{state}" [label={label}];\n')

            # Add terminal states with double rectangles
            for terminal_state in self.terminal_states:
                instrs = state_to_instrs.get(terminal_state, [])
                label = create_html_label(terminal_state, instrs)
                f.write(f'    "{terminal_state}" [label={label}, peripheries=1];\n')

            # Add START node with empty instruction lists
            f.write(f'    "START" [label={create_html_label("START", [])}];\n')

            f.write("}\n")
        logging.info("Graph written successfully")


def main() -> None:
    """Main entry point for the interpreter."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python " + sys.argv[0] + " <binary_path> <output_path> [<function_name>]")
        sys.exit(1)

    binary_path = sys.argv[1]
    output_path = sys.argv[2]
    function_name = sys.argv[3] if len(sys.argv) > 3 else "main"

    # Configure logging
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logging.info("Starting analysis of %s", binary_path)

    try:
        interpreter = Interpreter(binary_path, function_name)
        logging.info("Doing analysis pass 1/2")
        interpreter.prepare(False)
        interpreter.run()
        if interpreter.address_score == {}:
            raise ValueError("Failed to guess dispatch state address")
        interpreter.dispatch_state_addr = max(interpreter.address_score, key=interpreter.address_score.get)
        logging.info("Guessed dispatch state address: %s", hex(interpreter.dispatch_state_addr))
        logging.info("Doing analysis pass 2/2")
        interpreter.prepare(True)
        interpreter.run()
        interpreter.write_output(output_path)
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error("Analysis failed: %s", str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
