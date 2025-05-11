#!/usr/bin/env python3
import sys
import logging
import time
from typing import Dict, Set, List, Tuple, Optional
import domain
from miasm.ir.ir import AssignBlock
from miasm.expression.expression import LocKey
from miasm.analysis.machine import Machine
from miasm.analysis.binary import Container
from miasm.core.locationdb import LocationDB
import traceback
from config import LOG_LEVEL, LOG_FORMAT, TRACKED_DATA_SIZE, ESTIMATOR_RATIO, MAX_STATE_PER_INST
from utils import TOP, Address, Value, AbstractValue
from estimator import estimate, Path
import numpy as np
from immutabledict import immutabledict
from miasm.core.graph import DiGraph

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
        self.dispatch_state_addr = None
        self.seeing = set()
        self.seen = set()
        self.loops = self.ir_cfg.compute_natural_loops(self.entry_block)
        self.loop_header = next(self.loops)[0][1]

    def prepare(self, state_split: bool) -> None:
        # Dispatch state management
        self.state_split = state_split
        self.node_state = {}                
        self.trace = None
        self.samples = []
        self.best = None
        self.best_2nd = None
        self.best_addr = None
        self.best_2nd_addr = None        
        self.common_vars = None

        # Initialize analysis state
        self.abstract_states: Dict[StateKey, domain.KSetsDomain] = {}
        self.worklist: List[Tuple[LocKey, DispatchState]] = [(self.entry_block, TOP)]
        self.abstract_states[(self.entry_block.key, TOP)] = self.domain_class(False)
        self.current_iter = -1
        self.iter_freshness = {}
        
        # Initialize graph structures
        self.splitted_graph: DiGraph = DiGraph()      

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
        self.asm_cfg = disasm_engine.dis_multiblock(entry_offset)
        lifter = machine.lifter_model_call(self.loc_db)
        self.ir_cfg = lifter.new_ircfg_from_asmcfg(self.asm_cfg)
        self.entry_block = self.ir_cfg.get_block(entry_loc).loc_key

    def _update_node_state(self, node: LocKey, input_state: AbstractValue):
        """Update the instruction-to-state mapping"""
        if self.state_split and input_state is not TOP and len(input_state) == 1:
            if node.key not in self.node_state:
                self.node_state[node.key] = set()
            self.node_state[node.key].add(next(iter(input_state)))


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

        if current_block == self.loop_header:
            self.current_iter += 1
        
        logging.debug("\nProcessing block at %s (address: %s)", 
                     current_block, hex(block_offset) if block_offset is not None else "<synthetic block>")
        
        # Read input dispatch state
        input_state: AbstractValue = abs_state.read(self.dispatch_state_addr, TRACKED_DATA_SIZE) if self.state_split else TOP         

        logging.debug("Input dispatch state: %s", 
                     str([hex(val) for val in input_state]) if input_state is not TOP else "TOP")
            
        self._update_node_state(current_block, input_state)
        # Process all assignments in the block        
        for assignment in ir_block.assignblks:            
            abs_state.update(assignment)
                 
        # Read output dispatch state
        output_state: AbstractValue = abs_state.read(self.dispatch_state_addr, TRACKED_DATA_SIZE) if self.state_split else TOP
        logging.debug("Output dispatch state: %s", 
                     str([hex(val) for val in output_state]) if output_state is not TOP else "TOP")
        
        return input_state, output_state

    def _process_successors(self, current_block: LocKey, abs_state: domain.KSetsDomain, input_state: AbstractValue, output_state: AbstractValue) -> None:
        """Process all successor blocks of the current location.
        
        Args:
            current_block: Current block being processed
            abs_state: Current abstract state
            input_state: Input dispatch state of current block
            output_state: Output dispatch state from current block
        """
        cur_offset = self.loc_db.get_location_offset(current_block)
        cur_addr_str = hex(cur_offset) if cur_offset is not None else "<synthetic block>"          
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
                source = (current_block, next(iter(input_state)) if input_state is not None else None)                
                sink = (succ_block, next_state)            
                self.splitted_graph.add_uniq_edge(source, sink)
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
            if self.state_split or (succ_block != self.loop_header and self.iter_freshness.get(succ_block.key, -1) == self.current_iter):
                succ_state.join(self.abstract_states[state_key])
            self.iter_freshness[succ_block.key] = self.current_iter
            
        if state_key not in self.abstract_states or not succ_state.equals(self.abstract_states[state_key]):
            self.abstract_states[state_key] = succ_state
            self.worklist.append((succ_block, next_state))
            logging.debug("Scheduling successor %s with state %s", 
                         succ_addr_str, hex(next_state) if next_state is not None else "TOP")
        else:
            logging.debug("Skipping already processed state %s at %s", 
                         hex(next_state) if next_state is not None else "TOP", succ_addr_str)

    def _record_trace(self, current_block: LocKey, abs_state: domain.KSetsDomain) -> None:
        """Record trace during pass 1 to guess state-split criterion"""
        if self.trace is not None:
            self.trace["path"].append((current_block.key, immutabledict(abs_state.get_static_vars())))
        if self.loop_header == current_block:
            if self.trace is not None:
                vars = set(self.trace["mem"].keys())
                if self.common_vars is None:
                    self.common_vars = vars
                else:
                    self.common_vars = self.common_vars.intersection(vars)    
                
                # Enable score-caching by path reference
                self.trace["path"] = Path(self.trace["path"])
                self.samples.append(immutabledict(self.trace))                          
            self.trace = {
                "path": []
            }   

            self.trace["mem"] = immutabledict(abs_state.get_static_vars())

    def _estimate(self) -> bool:
        """Estimate the state-split address, returns True if estimation is stable"""
        t = estimate(self.samples, self.common_vars)
        if t is not None:

            (self.best, self.best_addr), (self.best_2nd, self.best_2nd_addr) = t
            candidate = None
            if self.best is not None and self.best_2nd != 0.0 and (self.best == 0.0 or (self.best_2nd / self.best) > ESTIMATOR_RATIO):            
                candidate = self.best_addr            
            
            decision = self.dispatch_state_addr is not None and candidate is not None and self.dispatch_state_addr == candidate
            self.dispatch_state_addr = candidate
            return decision
           
    def run(self) -> None:
        """Run the analysis until all reachable states are processed."""
        logging.info("Starting analysis...")
        steps = 0
        while self.worklist:
            steps += 1
            current_block, current_state = self.worklist.pop()            
            abs_state = self.abstract_states.get((current_block.key, current_state), self.domain_class(True)).clone()

            if not self.state_split:
                self._record_trace(current_block, abs_state)     
                if self.loop_header == current_block:
                    if self._estimate():
                        break
          
            input_state, output_state = self._process_block(current_block, abs_state)            
            
            if not abs_state.is_bot:
                self._process_successors(current_block, abs_state, input_state, output_state)
        logging.info("Abstract interpretation converged in " + str(steps) + " steps.")

    def _remove_nodes(self, remove: Set) -> None:
        """Remove nodes in set from graph"""        
        
        for node in remove:
            bypass_edges = set()
            # Bypass node
            for succ in self.splitted_graph.successors_iter(node):
                for pred in self.splitted_graph.predecessors_iter(node):
                        bypass_edges.add((pred, succ))

            for (pred, succ) in bypass_edges:
                self.splitted_graph.add_edge(pred, succ)           

            # Remove node
            self.splitted_graph.del_node(node)
               
    def postprocess(self) -> None:
        """Postprocess graph to remove dispatcher and redundant nodes"""
        self.node_blocks = {}
        for node in self.splitted_graph.nodes():
            block = self.asm_cfg.loc_key_to_block(node[0]) 
            self.node_blocks[node] = [block]

        # Remove dispatcher nodes
        remove = set()
        for node in self.splitted_graph.nodes():
            if self.asm_cfg.loc_key_to_block(node[0]) is None or node[1] is not None and (node[0].key not in self.node_state or len(self.node_state[node[0].key]) > MAX_STATE_PER_INST):
                remove.add(node) 
        self._remove_nodes(remove)      
            
        # Merge redundant nodes
        m = 0
        change = True
        while change:
            change = False
            remove = set()
            for node in self.splitted_graph.nodes():
                if len(self.splitted_graph.successors(node)) == 1:
                    succ = next(self.splitted_graph.successors_iter(node))
                    if len(self.splitted_graph.predecessors(succ)) == 1:
                        # Merge code of removed node                   
                        self.node_blocks[node] += self.node_blocks[succ]
                        change = True                        
                        self._remove_nodes(set([succ]))
                        break

    def write_output(self, output_path: str) -> None:
        """Write the state transition graph to a DOT file.
        
        Args:
            output_path: Path to write the DOT file
        """
        logging.info("Writing state transition graph to %s", output_path)
        with open(output_path, "w") as f:
            f.write("digraph finite_state_machine {\n")
            f.write("    rankdir=TB;\n")  # Changed from LR to TB for top-to-bottom layout
            f.write("    node [shape=none, fontname=\"Courier\"];\n")
            f.write("    edge [fontname=\"Courier\"];\n")
            f.write("    nodesep=0.5;\n")  # Add some horizontal spacing between nodes
            f.write("    ranksep=0.5;\n")  # Add some vertical spacing between ranks

            # Write edges
            
            node_label = lambda node : ("loc_key_" + str(node[0].key), "state_" + hex(node[1]) if node[1] is not None else "START")
            for source in self.splitted_graph.nodes():
                for target in self.splitted_graph.successors_iter(source):
                    f.write(f'    "{node_label(source)}" -> "{node_label(target)}";\n')
     
            def create_html_label(state, blocks):
                html = ['<']
                html.append('<TABLE CELLBORDER="0" CELLSPACING="0">')
                html.append(f'<TR><TD COLSPAN="2" ALIGN="CENTER"><B>{"loc_key_" + str(state[0].key) + ", state="+("START" if state[1] is None else hex(state[1]))}</B></TD></TR>')
                for block in blocks:
                    for instr in block.lines:
                        addr_str = f"0x{hex(instr.offset)[2:].zfill(8)}"
                        disasm = str(instr)
                        html.append(f'<TR><TD ALIGN="RIGHT">{addr_str}:</TD><TD ALIGN="LEFT">{disasm}</TD></TR>')
                html.append('</TABLE>')
                html.append('>')
                return ''.join(html)

            # Write nodes with instruction information
            for source in self.splitted_graph.nodes():                               
                label = create_html_label(source, self.node_blocks[source])              
                f.write(f'    "{node_label(source)}" [label={label}];\n')

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
        start_time = time.time()
        logging.info("Doing analysis pass 1/2")
        interpreter.prepare(False)
        interpreter.run()                
        logging.info("Guessed state-split criterion: %s", hex(interpreter.dispatch_state_addr))
        logging.info("Doing analysis pass 2/2")
        interpreter.prepare(True)
        interpreter.run()
        interpreter.postprocess()
        interpreter.write_output(output_path)
        logging.info(f"Analysis completed successfully in {time.time() - start_time:.3f} sec(s)")
    except Exception as e:
        logging.error("Analysis failed: %s", str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
