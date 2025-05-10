import logging

#arbitrary but realistic stack top addr
STACK_TOP_ADDR = 0x7ffde694fff8

#size of tracked data (track only data that may be the dispatcher state, for performance reasons)
TRACKED_DATA_SIZE = 4 

#max possible memory operation size on this architecture (if wrong, analysis will be unsound)
MAX_MEM_SIZE = 8 

#max kset size
KSET_MAX = 5

#Stops phase 1 when the ratio between best and second-best guess is higher than this
ESTIMATOR_RATIO = 2

#Weight of memory state during path similarity comparison
MEM_WEIGHT = 0.5

# Show on the CFG instructions that matches up to MAX_STATE_PER_INST states
MAX_STATE_PER_INST = 2

#log configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '[%(levelname)s] %(message)s'

