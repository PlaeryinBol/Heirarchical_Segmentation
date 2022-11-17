# dirs & files
IMAGE_DIR = './data/Pascal-part/JPEGImages'
MASK_DIR = './data/Pascal-part/gt_masks'
MODELS_DIR = './models'
TRAIN_IDS = './data/Pascal-part/train_id.txt'
VAL_IDS = './data/Pascal-part/val_id.txt'

# data
TEST_SIZE = 0.42
CLASSES = {
            0: 'bg',
            1: 'low_hand',
            2: 'torso',
            3: 'low_leg',
            4: 'head',
            5: 'up_leg',
            6: 'up_hand'
            }
TOP_LEVEL = (1, 2, 3, 4, 5, 6)  # all body
MIDDLE_LEVEL = (1, 2, 4, 6)  # hands, torso, head
BOTTOM_LEVEL = (3, 5)  # legs

# params
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 512
BATCH_SIZE = 7
MAX_LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4
NOT_IMPROVE = 8
