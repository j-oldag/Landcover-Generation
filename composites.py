# Resize and stack bands from multiple raster files into single image.

import tifffile as tiff
import numpy as np
import os
import shutil
from numpy.lib.stride_tricks import as_strided

# Receive argument from execution.py to create strings with paths.
# STATE MUST BE MANUALLY SET HERE, DUE TO NATURE OF POOL MULTIPROCESSING
def getDirs(county):
    county_path = os.path.join('C:/DeepUNET/geography/Georgia',county)
    getDirs.cDir = os.path.join(county_path,'HC')
    getDirs.nDir = os.path.join(county_path,'HN')
    getDirs.outDir = os.path.join(county_path,'Composites')
    if os.path.isdir(getDirs.outDir) == False:
        os.makedirs(getDirs.outDir)
    else:
        pass

# Create a list of all images from the county.
def imgList(directory):
    imgList.tifList = []
    for f in os.listdir(directory):
        if f.endswith(".TIF"):
            imgList.tifList.append(f)
    print('Directory: {} \n'.format(directory), len(imgList.tifList), 'images found')

# Seperate bands from original 3-band structures.
def seperate(nRast,cRast):
    seperate.red = nRast[:,:,0]
    seperate.blue = nRast[:,:,1]
    seperate.green = nRast[:,:,2]
    seperate.nir = cRast[:,:,0]

    seperate.red = seperate.red.astype(np.float16)
    seperate.blue = seperate.blue.astype(np.uint8)
    seperate.green = seperate.green.astype(np.uint8)
    seperate.nir = seperate.nir.astype(np.float16)

# Quick check to make sure that all bands have the same size, bounds, and resolution.
def resize(Red, Blue, Green, NIR):
    # print('Resizing Images')
    if Red.shape[0] == NIR.shape[0] and Red.shape[1] == NIR.shape[1]:
        resize.red = Red
        resize.blue = Blue
        resize.green = Green
        resize.nir = NIR
    if Red.shape[0] > NIR.shape[0] and Red.shape[1] > NIR.shape[1]:
        pad1 = Red.shape[0]-NIR.shape[0]
        pad2 = Red.shape[1]-NIR.shape[1]
        resize.nir = np.pad(NIR,((0,pad1),(0,pad2)),mode='edge')
        resize.red = Red
        resize.blue = Blue
        resize.green = Green
    #N is larger in neither vert or hor
    elif Red.shape[0] < NIR.shape[0] and Red.shape[1] < NIR.shape[1]:
        pad1 = NIR.shape[0]-Red.shape[0]
        pad2 = NIR.shape[1]-Red.shape[1]
        resize.red = np.pad(Red,((0,pad1),(0,pad2)),mode='edge')
        resize.blue = np.pad(Blue,((0,pad1),(0,pad2)),mode='edge')
        resize.green = np.pad(Green,((0,pad1),(0,pad2)),mode='edge')
        resize.nir = NIR
    #N is larger in hor NOT vert
    elif Red.shape[0] < NIR.shape[0] and Red.shape[1] > NIR.shape[1]:
        pad1 = NIR.shape[0]-Red.shape[0]
        pad2 = Red.shape[1]-NIR.shape[1]
        resize.red = np.pad(Red,((0,pad1),(0,0)),mode='edge')
        resize.blue = np.pad(Blue,((0,pad1),(0,0)),mode='edge')
        resize.green = np.pad(Green,((0,pad1),(0,0)),mode='edge')
        resize.nir = np.pad(NIR,((0,0),(0,pad2)),mode='edge')
    #N is larger in vert NOT hor
    elif Red.shape[0] > NIR.shape[0] and Red.shape[1] < NIR.shape[1]:
        pad1 = Red.shape[0]-NIR.shape[0]
        pad2 = NIR.shape[1]-Red.shape[1]
        resize.red = np.pad(Red,((0,0),(0,pad2)),mode='edge')
        resize.blue = np.pad(Blue,((0,0),(0,pad2)),mode='edge')
        resize.green = np.pad(Green,((0,0),(0,pad2)),mode='edge')
        resize.nir = np.pad(NIR,((0,pad1),(0,0)),mode='edge')
    #Vert is equal, N is NOT larger in hor
    elif Red.shape[0] == NIR.shape[0] and Red.shape[1] < NIR.shape[1]:
        pad2 = NIR.shape[1]-Red.shape[1]
        resize.red = np.pad(Red,((0,0),(0,pad2)),mode='edge')
        resize.blue = np.pad(Blue,((0,0),(0,pad2)),mode='edge')
        resize.green = np.pad(Green,((0,0),(0,pad2)),mode='edge')
        resize.nir = NIR
    #Vert is equal, N is larger in hor
    elif Red.shape[0] == NIR.shape[0] and Red.shape[1] > NIR.shape[1]:
        pad2 = Red.shape[1]-NIR.shape[1]
        resize.nir = np.pad(NIR,((0,0),(0,pad2)),mode='edge')
        resize.red = Red
        resize.blue = Blue
        resize.green = Green
    #Hor is equal, N is NOT larger in vert
    elif Red.shape[0] < NIR.shape[0] and Red.shape[1] == NIR.shape[1]:
        pad1 = NIR.shape[0]-Red.shape[0]
        resize.red = np.pad(Red,((0,pad1),(0,0)),mode='edge')
        resize.blue = np.pad(Blue,((0,pad1),(0,0)),mode='edge')
        resize.green = np.pad(Green,((0,pad1),(0,0)),mode='edge')
        resize.nir = NIR
    #Hor is equal, N is larger in vert
    elif Red.shape[0] > NIR.shape[0] and Red.shape[1] == NIR.shape[1]:
        pad1 = Red.shape[0]-NIR.shape[0]
        resize.nir = np.pad(NIR,((0,pad1),(0,0)),mode='edge')
        resize.red = Red
        resize.blue = Blue
        resize.green = Green

# Index creation, to be used as UNET input.
def NDVI(Red,NIR):
    # print('Calculating NDVI')
    np.seterr(divide='ignore',invalid='ignore')
    NDVI.ndvi = (NIR - Red)/(NIR + Red)
    NDVI.ndvi = np.nan_to_num(NDVI.ndvi)
    NDVI.ndvi = (NDVI.ndvi * 255).round().astype(np.uint)

# Index creation, to be used as UNET input.
def EVI(Red, NIR):
    # print('Calculating EVI')
    EVI.evi = (2.5*(NIR - Red)/(NIR + 2.4 * Red + 1.0))
    EVI.evi = np.nan_to_num(EVI.evi)
    EVI.evi = (EVI.evi * 255).round().astype(np.uint8)

# Band creation and smoothing, to be used for index creation.
def NIRrange(NIR):
    # print('Deriving NIR range')
    m, n = NIR.shape
    newshape = (m-7+1, n-7+1, 7, 7)
    newstrides = NIR.strides * 2
    NIR_max = as_strided(NIR, newshape, newstrides).max(axis=(2,3))
    NIR_min = as_strided(NIR, newshape, newstrides).min(axis=(2,3))
    NIRrange.range = NIR_max-NIR_min
    NIRrange.range = np.pad(NIRrange.range,((3,3),(3,3)),mode='edge')

# Band creation and smoothing, to be used for index creation.
def greenRange(Green):
    # print('Deriving green range')
    m, n = Green.shape
    newshape = (m-7+1, n-7+1, 7, 7)
    newstrides = Green.strides * 2  # strides is a tuple
    Green_max = as_strided(Green, newshape, newstrides).max(axis=(2,3))
    Green_min = as_strided(Green, newshape, newstrides).min(axis=(2,3))
    greenRange.range = Green_max-Green_min
    greenRange.range = np.pad(greenRange.range,((3,3),(3,3)),mode='edge')

# Texture estimation to improve tree-crown detection
def texture(NIR_Range,Green_Range):
    # print('Calculating Texture')
    texture.text = ((NIR_Range + Green_Range)/2)
    m,n = texture.text.shape
    newshape = (m-3+1, n-3+1, 3, 3)
    newstrides = texture.text.strides * 2  # strides is a tuple
    texture.text = as_strided(texture.text, newshape, newstrides).mean(axis=(2,3))
    texture.text = np.pad(texture.text,((1,1),(1,1)),mode='edge')

# Stack all layers into single 7-band image.
def composite(Red, Green, Blue, NIR, NDVI, EVI, Texture, outDirectory, imgName):
    # print('Compositing bands')
    red = Red.astype(np.uint8)
    green = Green.astype(np.uint8)
    blue = Blue.astype(np.uint8)
    nir = NIR.astype(np.uint8)
    ndvi = NDVI.astype(np.uint8)
    evi = EVI.astype(np.uint8)
    texture = Texture.astype(np.uint8)

    Composite = np.dstack((red,green,blue,nir,ndvi,evi,texture)).transpose([2,0,1])
    dest_path = os.path.join(outDirectory,imgName)
    tiff.imsave(dest_path,Composite)
    print(dest_path)

# Copy meta and spatial data from input image to new composite.
def copyMeta(name,source,dest):
    tfw = '{}.tfw'.format(name)
    xml = '{}.TIF.aux.xml'.format(name)
    metaList = [tfw,xml]
    for m in metaList:
        src_path = os.path.join(source,m)
        dst_path = os.path.join(dest,m)
        shutil.copyfile(src_path,dst_path)

# Combines above helper functions to create composite image.
def generation(imageList, cDirectory, nDirectory):
    for i in imageList:
        imgName = os.path.splitext(i)[0]
        cPath = os.path.join(cDirectory,i)
        nPath = os.path.join(nDirectory,i)
        cRast = tiff.imread(cPath)
        nRast = tiff.imread(nPath)

        seperate(nRast,cRast)
        resize(seperate.red,seperate.blue,seperate.green,seperate.nir)
        NDVI(resize.red,resize.nir)
        EVI(resize.red,resize.nir)
        NIRrange(resize.nir)
        greenRange(resize.green)
        texture(NIRrange.range,greenRange.range)
        composite(resize.red,resize.green,resize.blue,resize.nir,NDVI.ndvi,EVI.evi,texture.text,getDirs.outDir,i)
        copyMeta(imgName,cDirectory,getDirs.outDir)

# Combines above helper functions to receive data location and create composite image.
def main(county):
    getDirs(county)
    imgList(getDirs.cDir)
    generation(imgList.tifList, getDirs.cDir, getDirs.nDir)
