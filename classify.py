# Module to classify all predictions and flatten predictions into single band.

import numpy as np
import tifffile as tiff
import rasterio
from rasterio.merge import merge
import os, shutil

# Receive argument from execution.py to create strings with paths.
def getDirs(county):
    dir_path = os.path.join('C:/DeepUNET/geography/Georgia',county)
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)
    else:
        pass
    getDirs.predDir = os.path.join(dir_path,'Predictions')
    getDirs.compDir = os.path.join(dir_path,'Composites')
    getDirs.classDir = os.path.join(dir_path,'Classified')
    if os.path.isdir(getDirs.classDir) == False:
        os.makedirs(getDirs.classDir)
    else:
        pass
    getDirs.flatDir = os.path.join(dir_path,'Flats')
    if os.path.isdir(getDirs.flatDir) == False:
        os.makedirs(getDirs.flatDir)
    else:
        pass
    getDirs.mergedDir = os.path.join(dir_path,'Merged')
    if os.path.isdir(getDirs.mergedDir) == False:
        os.makedirs(getDirs.mergedDir)
    else:
        pass

# Finds all images inside of the predictions directories.
def imgList(inputDirectory):
    testDir = os.path.join(inputDirectory,'00')
    imgList.list = []
    for f in os.listdir(testDir):
        if f.endswith(".TIF"):
            imgList.list.append(f)

# Feed a single image name, reads all 7 predictions and output 7 classified images.
def classify(image,predDirectory,subDirectory,threshold,assign):
    srcPath = os.path.join(predDirectory,image)
    dstPath = os.path.join(subDirectory,image)
    tif = tiff.imread(srcPath)
    tif[tif<threshold]=0
    tif[tif>=threshold]=assign
    tiff.imsave(dstPath,tif)

# Creates sub-directories for each classification in landcover schema.
def createSubFolders(inputDirectory):
    for i in range(0,7):
        dstDir = os.path.join(inputDirectory,'0{}'.format(i))
        if os.path.isdir(dstDir) == False:
            os.makedirs(dstDir)
        else:
            pass

# Combine all 7 classified landcover layers into single band.
# Order is determined by testing accuracy of each landcover class.
def flatten(image,classifiedDirectory,flattenedDirectory,compositesDirectory):
    print('Flattening',flattenedDirectory,image)
    orderList = ['02','01','04','05','06','00','03']
    for o in range(len(orderList)):
        srcPath = os.path.join(classifiedDirectory,orderList[o])
        srcPath = os.path.join(srcPath,image)
        if o == 0:
            carry = tiff.imread(srcPath)
            mask = (carry == 0)
        elif o == 6:
            temp = tiff.imread(srcPath)
            np.copyto(carry,temp,casting='same_kind',where=mask)
            carry[carry>7]=0
            flattenedPath = os.path.join(flattenedDirectory,image)
            tiff.imsave(flattenedPath,carry)
        else:
            temp = tiff.imread(srcPath)
            np.copyto(carry,temp,casting='same_kind',where=mask)
            mask = (carry == 0)
        os.unlink(srcPath)
    copyMeta(image,compositesDirectory,flattenedDirectory)

# Copy meta and spatial data from input imagery to classified image.
def copyMeta(fileName,source,dest):
    name = os.path.splitext(fileName)[0]
    tfw = '{}.tfw'.format(name)
    xml = '{}.TIF.aux.xml'.format(name)
    metaList = [tfw,xml]
    for m in metaList:
        src_path = os.path.join(source,m)
        dst_path = os.path.join(dest,m)
        shutil.copyfile(src_path,dst_path)

# Merge all classified tiles into county-wide landcover raster.
def merge(sourceDirectory,destinationDirectory,county,imageList):
    print('Merging',county)
    dstPath = os.path.join(destinationDirectory,'{}_Landcover.TIF'.format(county))
    src_files = []
    for i in imageList:
        srcPath = os.path.join(sourceDirectory,i)
        src = rasterio.open(srcPath)
        src_files.append(src)
    mosaic, out_trans = rasterio.merge.merge(src_files,method='max')
    out_meta=src.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],

                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    })
    with rasterio.open(dstPath, "w", **out_meta) as dest:
        dest.write(mosaic)

# Combine above helper functions to output final landcover classification.
# Also clean up the composite, prediction, and flatten layers create earlier.
def main(county):
    threshList = [180,40,140,100,35,100,70]
    getDirs(county)
    createSubFolders(getDirs.classDir)
    imgList(getDirs.predDir)
    for i in imgList.list:
        for n in range(0,7):
            predPath = os.path.join(getDirs.predDir,'0{}'.format(n))
            classPath = os.path.join(getDirs.classDir, '0{}'.format(n))
            print('Classifying',classPath)
            classify(i,predPath,classPath,threshList[n],n+1)
        flatten(i,getDirs.classDir,getDirs.flatDir,getDirs.compDir)
    shutil.rmtree(getDirs.classDir)
    merge(getDirs.flatDir,getDirs.mergedDir,county,imgList.list)
    shutil.rmtree(getDirs.flatDir)
    shutil.rmtree(getDirs.compDir)
    shutil.rmtree(getDirs.predDir)
