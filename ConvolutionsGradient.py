import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('./Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Convention OpenCV : une image de type entier est interprétée dans {0,...,255}


# Calcul de gradx


t1 = cv2.getTickCount()
# kernel = (1/2)*np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
# Masques de sobel
kernel = (1/8)*np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
gradx = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de gradx :",time,"s")

gradx_normalized = gradx
# gradx_normalized = gradx+122.5

print("Valeur maximale: ",np.max(gradx_normalized),"\nValeur minimale: ",np.min(gradx_normalized))
cv2.imshow('Gradient in the X axis',gradx_normalized/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.figure()
plt.imshow(gradx_normalized,cmap = 'gray',vmin = -255.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Gradient in the X axis')

plt.show()


# Calcul de grady


t1 = cv2.getTickCount()
# kernel = (1/2)*np.array([[0, 1, 0],[0, 0, 0],[0, -1, 0]])
# Masque de sobel
kernel = (1/8)*np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
grady = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de gradx :",time,"s")

grady_normalized = grady
#grady_normalized = grady+122.5

cv2.imshow('Gradient in the Y axis',grady_normalized/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.figure()
plt.imshow(grady_normalized,cmap = 'gray',vmin = -255.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()


# Calcul du laplacian


#Méthode filter2D
t1 = cv2.getTickCount()
# 4-connexité
kernel = (1/8)*np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
# 8-connexité
kernel = (1/16)*np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
laplacian = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul du laplacian :",time,"s")

laplacian_normalized = laplacian

cv2.imshow('Laplacian',laplacian_normalized/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.figure()
plt.imshow(laplacian_normalized,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Laplacian')

plt.show()