# Automated Unflattening via State-sensitive Program EXploration (AUSPEX)

This project demonstrates the use of abstract interpretation for binary analysis, specifically focusing on tracking dispatch state transitions in x86_64 binaries obfuscated by control flow flattening (CFF). It serves as both an educational resource and a proof-of-concept implementation.

<p align="center">
  <img src="https://raw.githubusercontent.com/klemmm/auspex/refs/heads/main/doc/schema.png" alt="CFG unflattening"/>
</p>

## Overview

The project implements an abstract interpreter that:
- Uses a k-sets abstract domain to track possible values
- Analyzes binary code to track dispatch state transitions
- Generates a state transition graph showing how different dispatch states are connected
- **Implements state splitting**: maintains separate abstract states for different dispatch values, allowing precise tracking of program behavior under different conditions
- **Provides static analysis guarantees**: sound analysis that covers all possible execution paths without requiring concrete execution

## Key Concepts

### Abstract Interpretation
Abstract interpretation is a static analysis technique that approximates program behavior by executing the program on abstract values rather than concrete ones. This provides several key advantages:

- **Soundness**: The analysis is guaranteed to cover all possible program behaviors - no execution path can be "missed"
- **Termination**: The analysis is guaranteed to terminate, even for programs with unbounded loops, as long as the dispatch state detection is successful
- **Efficiency**: The analysis does not need to explicitely trace each path

### K-Sets Domain
The k-sets domain is an abstract domain that tracks sets of concrete values, with a maximum size limit (k). It provides a lattice structure where:
- TOP represents all possible values
- Sets represent specific sets of possible values
- BOTTOM (empty set) represents no possible values

### Dispatch State Analysis
The analysis tracks how dispatch states (values at a specific memory address) change through program execution, building a graph of possible state transitions. This static analysis approach means:
- All possible state transitions are discovered
- No concrete execution is needed to explore different paths
- The analysis terminates even with unbounded loops
- Results are valid for all possible inputs

The dispatch state address is guessed by a preliminary analysis pass by detecting the variable that correlates the most with the path taken in each iteration. Heuristics parameters are configurable in `config.py`

### State Splitting
A key feature of this implementation is its ability to maintain separate abstract states for different dispatch values. When the analysis encounters a branch or state change:
- It creates a new abstract state for each possible dispatch value
- These abstract states are tracked independently through the program
- This allows precise analysis of program behavior under different conditions
- The state splitting mechanism is crucial for handling conditional branches and state-dependent behavior

### Detecting original code blocks and associating them to dispatcher state values

For each basic block, the associated abstract states are checked, if the abstract state contains only few (threshold configurable) possible dispatcher state value, then the instruction is considered belonging to original code, and matched to the corresponding dispatcher state value.

The obfuscation/dispatcher instructions usually corresponds to multiple dispatcher state values, and therefore are not considered original code instructions.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/klemmm/auspex.git
cd auspex
```

2. Create and activate a python virtual env (optional but recommended)
```bash
python3 -m venv env
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
./auspex.py examples/example graph.dot
```

This will:
- Load and analyze the specified binary
- Track dispatch state transitions
- Handle loops and recursion without concrete execution
- Detect instructions beloging to each dispatcher state by using the abstract interpretation results
- Generate a DOT file showing the state transition graph along with instructions for each node
- You can visualize the graph using Graphviz:
```bash
dot -Tpng graph.dot -o graph.png
```

## Example results 

### Example source code

```c
int main(int argc, char **argv) {
    unsigned int i;
    unsigned int ret = 0;
    for (i = 0; i < argc; i++) {
        if (argv[i][0] != 0) {
            ret++;
        }
    }
    return ret;
}
```

### Example flattened CFG

<p align="center">
  <img src="https://raw.githubusercontent.com/klemmm/auspex/refs/heads/main/examples/example-flattened.png" alt="example CFG unflattened"/>
</p>

### Example unflattened CFG

<p align="center">
  <img src="https://raw.githubusercontent.com/klemmm/auspex/refs/heads/main/examples/example.png" alt="example CFG unflattened"/>
</p>

## Configuration

The analysis can be configured through `config.py`:
- `KSET_MAX`: Maximum size of value sets in the abstract domain
- `LOG_LEVEL`: Logging verbosity

## Project Structure

- `domain.py`: Implements the k-sets abstract domain
- `auspex.py`: Main analysis engine
- `utils.py`: Utility functions and type definitions
- `config.py`: Configuration parameters
- `tests/`: Test suite
- `examples/`: Example binaries and usage

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Future works: towards handling de-virtualization

Running AUSPEX on a simple VM program (`examples/vm.c`) successfully detects the VIP address and recover the control flow of program emulated in the VM (note: modifying `MAX_STATE_PER_INST` to 2 in `config.py` is required): 

<p align="center">
  <img src="https://raw.githubusercontent.com/klemmm/auspex/refs/heads/main/examples/vm.png" alt="example VM graph"/>
</p>

## Limitations / TODO

This is an educational/PoC tool, it has limitations such as: 
- Not all IR instructions are handled
- Inter-procedural analysis is not done
- The abstract domain is quite simple (not relational) although it suffices for now 
- No support for stuff like opaque predicates, etc. 
- Should add an optimization pass to remove useless instructions (e.g. managing dispatcher state) from the final result
- Should ensure soundness for memory write with non-static addresses (needs a more powerful analysis, VSA / intervals / ...)
- Should handle better CFG construction (i.e. dynamic branches)
- Contents from .data/.rodata should be taken into account in initial abstract state
- Supports only *one* level of flattening / dispatcher state variable for now.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- Based on abstract interpretation concepts from Cousot & Cousot
- Uses the Miasm framework for binary analysis 
