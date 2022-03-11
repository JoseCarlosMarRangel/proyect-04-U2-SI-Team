from skimage import io
# from skimage.io import imshow, imread
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


# Prueba con una sola imagen
image_path_list = os.listdir('./piezas/')
i = 17
image_path = image_path_list[i]
# image = rgb2gray(io.imread('./dataset/'+image_path))

# Aplica un filtro en la imagen para que este en blanco y negro
# image_path = 'rey.jpg';
image = rgb2gray(io.imread('./piezas/'+image_path))
imageplt = io.imshow(image)
plt.show()

# Cambia la imagen para que tenga solo blanco y negro, quitando las sombras
binary = image < threshold_otsu(image)
binary = closing(binary)
binimg = io.imshow(binary)
plt.show()

# Label para seleccionar las regiones y separarlas por colores
label_img = label(binary)
io.imshow(label_img)
plt.show()

# Tabla con los datos de la imagen
table = pd.DataFrame(regionprops_table(label_img, image,
                                       ['convex_area', 'area',
                                        'eccentricity', 'extent',
                                        'inertia_tensor',
                                        'major_axis_length',
                                        'minor_axis_length']))
table['convex_ratio'] = table['area']/table['convex_area']
table['label'] = image_path[5]
print("Tabla de los datos de la imagen:")
print(table)

# exit()

# Prueba de implementacion de lo previo con todas las imagenes
# Por algun motivo que no entendemos presenta un index error, a pesar de que con anterioridad no pasaba
print("Todas las imagenes \n")
print("No cierre la ejecucion, se esta trabajando")
image_path_list = os.listdir('./piezas/')
df = pd.DataFrame()

for i in range(len(image_path_list)):
    image_path = image_path_list[i]
    image = rgb2gray(io.imread('./piezas/'+image_path))
    binary = image < threshold_otsu(image)
    binary = closing(binary)
    label_img = label(binary)

    table = pd.DataFrame(regionprops_table(label_img, image,
                                           ['convex_area', 'area',
                                            'eccentricity', 'extent',
                                            'inertia_tensor',
                                            'major_axis_length',
                                            'minor_axis_length']))
    table['convex_ratio'] = table['area']/table['convex_area']
    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []
    for prop in regionprops(label_img):

        min_row, min_col, max_row, max_col = prop.bbox
        img = image[min_row:max_row, min_col:max_col]
        real_images += [img]
        mean += [np.mean(img)]
        std += [np.std(img)]
        percent25 += [np.percentile(img, 25)]
        percent75 += [np.percentile(img, 75)]
    table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']
    table['label'] = image_path[5]
    df = pd.concat([df, table], axis=0)

df.head()
print(df)

X = df.drop(['label'], axis=1)

# Features o características
X = X[['iqr', 'real_images', 'mean_intensity',
       'std_intensity', '25th Percentile', '75th Percentile']]


# Targets
y = df['label']
columns = X.columns

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y)

clf = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, random_state=123)

clf.fit(X_train, y_train)  # print confusion matrix of test set
# print accuracy score of the test set
print(classification_report(clf.predict(X_test), y_test))
final = {np.mean(clf.predict(X_test) ==
                 y_test)*100: .2}
print("Test Accuracy: " + str(final) + "%")

exit()
