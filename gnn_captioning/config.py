MAX_LENGTH = 30
BATCH_SIZE = 64
device = "cuda"
flicker_path = "./Images"
flicker_caption = "./captions.txt"

LEARNING_RATE = 2e-4
train_coco_dir = "./train2014"
val_coco_dir = "./val2014"
coco_train_annos = "./annotations/captions_train2014.json"
coco_val_annos = "./annotations/captions_val2014.json"
flicker_ratio = (0.8, 0.2, 0.0)

NUM_PATCHES_VIT = 16
HIDDEN_DIM_VIT = 128
SIN_PE_VIT = True
DEPTH_VIT = 12
HEADS_VIT = 8

NUM_PATCHES_GCN = 64
HIDDEN_DIM_GCN = 128
SIN_PE_GCN = True
ENCODE_DIM_GCN = 128
GCN_LAYER = 3

NUM_PATCHES_GAT = 64
HIDDEN_DIM_GAT = 128
SIN_PE_GAT = True
ENCODE_DIM_GAT = 128
GAT_LAYER = 3

EPOCHS = 20