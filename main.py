from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import resize
from tqdm import tqdm
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk 
import os


# Prueba con una sola imagen
image_path_list = os.listdir('./dataset/')
i = 24
image_path = 'tBlackTowerFour.jpg'
# image = rgb2gray(io.imread('./dataset/'+image_path))

# Aplica un filtro en la imagen para que este en blanco y negro
# image_path = 'rey.jpg';
image = rgb2gray(io.imread('./dataset/tBlackTowerFour.jpg'))
##image = resize(image,(480,640), anti_aliasing = True)
imageplt = io.imshow(image)
plt.show()

# Cambia la imagen para que tenga solo blanco y negro, quitando las sombras
binary = image < threshold_otsu(image)
binary = closing(binary)
binimg = io.imshow(binary)
plt.show()


footprint = disk(6)
eroded = erosion(binary, footprint)

io.imshow(eroded)
plt.show()


dilated = dilation(eroded, footprint)
io.imshow(dilated)
plt.show()

# Label para seleccionar las regiones y separarlas por colores
label_img = label(dilated)
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
std = []
mean = []
percent25 = []
percent75 = []
for prop in regionprops(label_img):

    min_row, min_col, max_row, max_col = prop.bbox
    img = image[min_row:max_row, min_col:max_col]
    mean += [np.mean(img)]
    std += [np.std(img)]
    percent25 += [np.percentile(img, 25)]
    percent75 += [np.percentile(img, 75)]
table['mean_intensity'] = mean
table['std_intensity'] = std
table['25th Percentile'] = mean
table['75th Percentile'] = std
table['iqr'] = table['75th Percentile'] - table['25th Percentile']
table['label'] = image_path[0] + image_path[1]
print("Tabla de los datos de la imagen:")
print(table)
imagetest = table.drop(['label'], axis = 1)
imagetestlbl = table['label']
# exit()

# Prueba de implementacion de lo previo con todas las imagenes
# Por algun motivo que no entendemos presenta un index error, a pesar de que con anterioridad no pasaba
print("Todas las imagenes \n")
print("No cierre la ejecucion, se esta trabajando")
image_path_list = os.listdir('./dataset/')
df = pd.DataFrame()

for i in range(len(image_path_list)):
    image_path = image_path_list[i]
    print("Analizando imagen: " + image_path)
    image = rgb2gray(io.imread('./dataset/'+image_path))
    ##image = resize(image,(640,480), anti_aliasing = True)
    binary = image < threshold_otsu(image)
    binary = closing(binary)
    footprint = disk(6)
    eroded = erosion(binary, footprint)
    dilated = dilation(eroded, footprint)
    label_img = label(dilated)






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
    table['label'] = image_path[0] + image_path[1]
    df = pd.concat([df, table], axis=0)

df.head()
print(df)

X = df.drop(["label", "real_images"], axis=1)
y = df["label"]

columns = X.columns

# train-test-split
##X_train, X_test, y_train, y_test = train_test_split(
    ##X, y, test_size=0.25, random_state=123, stratify=y)
X_train, X_test, y_train, y_test = X,X,y,y
clf = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, random_state=123)

clf.fit(X_train, y_train)  # print confusion matrix of test set
# print accuracy score of the test set
print(classification_report(clf.predict(X_test), y_test))
final = {np.mean(clf.predict(X_test) ==
                 y_test)*100}
print("Test Accuracy: " + str(final) + "%")
#from collections import Counter
#mylist = clf.predict(imagetest)
#c = Counter(mylist)
#print(c.most_common(1), imagetestlbl.iloc[0])
