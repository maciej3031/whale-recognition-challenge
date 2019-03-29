import sys
import os
from skimage import io
from tqdm import tqdm
import numpy as np


ratios = []

for image in tqdm(os.listdir(sys.argv[1])):
    print("Processing image: {}".format(image))
    if 'DS_Store' in image:
        continue
    else: 
        img = io.imread(os.path.join(os.path.join(sys.argv[1],image)))
        shape =  img.shape
        height, width = shape[0], shape[1]

        aspect_ratio = width/height
        ratios.append(aspect_ratio)

print('ratios:', ratios)
print('mean ratios:', np.mean(ratios))


### USAGE: python aspect_ratio.py catalog_with_images_to_calculate
