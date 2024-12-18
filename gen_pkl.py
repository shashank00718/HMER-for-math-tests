'''
Originally written by Hongyu Wang in Beihang university

Code modified for academic use by Taanvi Dande of BITS Pilani, Hyd Campus.
'''
import os
import sys
import pandas as pd
import pickle as pkl
import numpy as np
from imageio import imread
from PIL import Image

image_path = r"C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\off_image_test\\"
outFile = 'offline-test.pkl'
oupFp_feature = open(outFile, 'wb')

features = {}
channels = 1
sentNum = 0

scpFile = open(r"C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\test_caption.txt")

while True:
    line = scpFile.readline().strip()
    if not line:
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '_0.bmp'
        
        try:
            im = imread(image_file)
            mat = np.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
            
            for channel in range(channels):
                image_file = image_path + key + f'_{channel}.bmp'
                im = imread(image_file)
                mat[channel, :, :] = np.array(im)
            
            sentNum += 1
            features[key] = mat
            
            if sentNum % 500 == 0:
                print('Processed sentences:', sentNum)
        
        except FileNotFoundError:
            print(f"Warning: File not found for key: {key}")
            continue

print('Image loading done. Total sentence number:', sentNum)

pkl.dump(features, oupFp_feature)
print('Save file done.')

oupFp_feature.close()
scpFile.close()
