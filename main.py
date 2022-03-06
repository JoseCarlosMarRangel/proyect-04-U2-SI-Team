from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
import csv
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import os


#img = io.imread('./dataset/BlackBishop(1).jpg')
# io.imshow(img)
# io.show()


# * el dataset se muestra en la terminal como una matriz casi casi
print(os.listdir('./dataset/'))
path_image = './dataset/'
image = io.imread(path_image + 'BlackBishop.jpg')
imageplt = plt.imshow(image)
plt.show()
exit()
