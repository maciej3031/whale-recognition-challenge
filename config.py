import os
from imgaug import augmenters as iaa
import imgaug as ia


DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpnts")
CONFIG_HISTORY_FILE = os.path.join(CHECKPOINTS_DIR, "config_history.csv")

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

SEQ = iaa.Sequential(
    [
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.9, 1.1),
                    "y": (0.9, 1.1),
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-15, 15),  # rotate by -45 to +45 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=['constant', 'edge']
            )
        ),
        # crop images by -5% to 10% of their height/width
        sometimes(
            iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))
        ),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.OneOf(
            [
                iaa.GaussianBlur(
                    (0, 1.0)
                ),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 2)),
                # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 3)),
                # blur image using local medians with kernel sizes between 2 and 7
                iaa.Emboss(alpha=(0, 1.0), strength=(0.2, 1.5)),  # emboss images
            ]
        ),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # add gaussian noise to images
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        # either change the brightness of the whole image (sometimes
        # per channel) or change the brightness of subareas
        iaa.Add(
            (-10, 10), per_channel=0.5
        ),  # change brightness of images (by -10 to 10 of original value)
        # iaa.Scale({"height": 32, "width": 64})
    ],
    random_order=True,
)

INPUT_SHAPE = (320, 320, 3)
VALID_IMG_NUM = 6000
BATCH_SIZE = 8
GREY_SCALE = True
USE_CROP = True

DEFAULT_HARD_SAMPLING_BATCH = 100
DEFAULT_MODEL_NAME = 'vgg19_fc_l2'
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_LOAD_WEIGHTS = False
DEFAULT_FREEZE_LAYERS = None
DEFAULT_MARGIN = 0.5
DEFAULT_TRAINING_CONFIG = None
DEFAULT_TIMESTAMP = None
USE_SIAMESE_MODEL=False

MAX_TRAIN_STEPS = 100
MAX_TRAIN_EPOCHS = 50

GPU_MEMORY_FRAC_TO_USE = 1.0

# Batch hard parameters:
DEFAULT_USE_HARD_BATCH = True
P = 30
# number of samples per class
K = 4
