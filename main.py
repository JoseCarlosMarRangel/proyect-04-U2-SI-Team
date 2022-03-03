from skimage import io
import numpy as np
import matplotlib.pyplot as plt

img = io.imread('./dataset/BlackBishop(1).jpg')
io.imshow(img)
io.show()
