EXP_NAME = "CIFAR10"

NUM_FEATURES = 3 * 32 * 32
NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000

NUM_CLIENTS = 10
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 20
INIT_LR = 0.1

EVAL_DISP_INTERVAL = 10

# VGG-11
DENSE_TIME = 31.514276721399803
SPARSE_ALL_TIME = 31.08082229309948
SPARSE_TIME = 17.36990105989628

COEFFICIENTS_SINGLE = [0., 8.95507e-6, 2.495288e-6, 2.780686e-6, 1.024265e-6, 1.277773e-6, 1.843831e-6,
                       8.066104e-7, 5.145334e-7, 2.430023e-7, 0.]
COMP_COEFFICIENTS = [c * NUM_LOCAL_UPDATES for c in COEFFICIENTS_SINGLE]
# 1MBps = 4e-6 * 2
COMM_COEFFICIENT = 5.561621025626998e-06  # 5.18405073e-6
TIME_CONSTANT = SPARSE_TIME * NUM_LOCAL_UPDATES

MAX_ROUND = 10001

# Adaptive pruning config
ADJ_INTERVAL = 50

IP_MAX_ROUNDS = 1000
IP_ADJ_INTERVAL = ADJ_INTERVAL
IP_DATA_BATCH = 10
IP_THR = 0.1

ADJ_THR_FACTOR = 1.5
ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES

# Variables
MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

LR_HALF_LIFE = 10000
ADJ_HALF_LIFE = 10000

# Iterative pruning config
NUM_ITERATIVE_PRUNING = 20

# Online algorithm config
MAX_NUM_UPLOAD = 5

DATA_SPLIT_MODE = 'non-iid'
NON_IID_N = 2
SHUFFLE = {'train': True, 'test': False}
BATCH_SIZE = 128