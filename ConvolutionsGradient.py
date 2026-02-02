import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('./Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
gradx = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = (0.5)*img[y, x+1] - (0.5)*img[y, x-1] 
    gradx[y,x] = val
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

gradx_min0 = gradx+122.5

cv2.imshow('Avec boucle python',gradx_min0.astype(np.uint8))
#Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
cv2.waitKey(0)

plt.subplot(121)
plt.imshow(gradx_min0,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, 0, 0],[-1/2, 0, 1/2],[0, 0, 0]])
gradx_2 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

gradx_2_min0 = gradx_2+122.5

cv2.imshow('Avec filter2D',gradx_2_min0/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.subplot(122)
plt.imshow(gradx_2_min0,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()

#Méthode directe
t1 = cv2.getTickCount()
grady = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = (0.5)*img[y+1, x] - (0.5)*img[y-1, x] 
    grady[y,x] = val
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

grady_min0 = grady+122.5

cv2.imshow('Avec boucle python',grady_min0.astype(np.uint8))
#Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
cv2.waitKey(0)

plt.subplot(121)
plt.imshow(grady_min0,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, 1/2, 0],[0, 0, 0],[0, -1/2, 0]])
grady_2 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

grady_2_min0 = grady_2+122.5

cv2.imshow('Avec filter2D',grady_2_min0/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.subplot(122)
plt.imshow(grady_2_min0,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()