#Prediction module

import math
import numpy as np
import tifffile as tiff
import os
import shutil
from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES

# Receive argument from execution.py to create strings with paths.
def getDirs(county):
    county_path = os.path.join('C:/DeepUNET/geography/Georgia',county)
    getDirs.compDir = os.path.join(county_path,'Composites')
    getDirs.outDir = os.path.join(county_path,'Predictions')
    if os.path.isdir(getDirs.outDir) == False:
        os.makedirs(getDirs.outDir)
    else:
        pass
    for i in range(0,7):
        predDir = os.path.join(getDirs.outDir,'0{}'.format(i))
        if os.path.isdir(predDir) == False:
            os.makedirs(predDir)
        else:
            pass

# Fast prediction of class using Deep UNET weights (Not provided).
def predict(x, model, patch_sz=320, n_classes=7):
    img_height=x.shape[0]
    img_width=x.shape[1]
    n_channels=x.shape[2]

    # Make extended img so that it contains integer number of patches.
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    if npatches_vertical == npatches_horizontal:
        pass
    elif npatches_vertical > npatches_horizontal:
        npatches_horizontal = npatches_vertical
    elif npatches_horizontal > npatches_vertical:
        npatches_vertical = npatches_horizontal
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float16)

    # Fill extended image with mirrors.
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # Assemble all patches in one array.
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    patches_array = np.asarray(patches_list)

    # Prediction.
    patches_predict = model.predict(patches_array, batch_size=12)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float16)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]

# Create a list of all images from the county.
def imgList(directory):
    imgList.tifList = []
    for f in os.listdir(directory):
        if f.endswith(".TIF"):
            imgList.tifList.append(f)
    print('Testing image listing:\n', len(imgList.tifList), 'images found')

# Combines above helper functions to predict confidence of landcover class.
def iteratePreds(compDirectory,outDirectory):
    model = get_model()
    model.load_weights(weights_path)
    imgList(compDirectory)
    for tif in imgList.tifList:
        tifPath = os.path.join(compDirectory,tif)
        print(tifPath)
        img = np.float16(tiff.imread(tifPath) / 255).transpose([1,2,0])
        try:
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            mymat = temp
            results = (255*mymat).astype('uint8')
            for i in range(0,results.shape[0]):
                outPath = os.path.join(outDirectory,'0%s/%s' % (i,tif))
                tiff.imsave(outPath, results[i,:,:])
            # os.unlink(tifPath)
        except:
            pass

# Combines above helper functions to receive data location and predict confidence of landcover class.
def main(county):
    getDirs(county)
    iteratePreds(getDirs.compDir,getDirs.outDir)
