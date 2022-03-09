from skimage import io
#from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
import csv
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os


#img = io.imread('./dataset/BlackBishop(1).jpg')
# io.imshow(img)
# io.show()


# * el dataset se muestra en la terminal como una matriz casi casi
#print(os.listdir('./dataset/'))
#path_image = './dataset/'
#image = io.imread(path_image + 'BlackBishop.jpg')
#imageplt = plt.imshow(image)
#plt.show()

image_path_list = os.listdir('./dataset/')
# looking at the first image
#i = 0
#image_path = image_path_list[i]
#image = rgb2gray(io.imread('./dataset/'+image_path))

#Aplica un filtro en la imagen para que este en blanco y negro
#image_path = 'rey.jpg';
image = rgb2gray(io.imread('./dataset/'+'BlackKingThree.jpg'))
imageplt = io.imshow(image)
plt.show()

#Cambia la imagen para que tenga solo blanco y negro, quitando las sombras
binary = image < threshold_otsu(image)
binary = closing(binary)
binimg = io.imshow(binary)
plt.show()

#Label para seleccionar las regiones y separarlas por colores
label_img = label(binary)
io.imshow(label_img)
plt.show()



exit()
