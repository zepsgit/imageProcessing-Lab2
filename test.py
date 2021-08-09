from skimage.filters import threshold_otsu
import cv2
I3=cv2.imread('img3.png',0)
th3=threshold_otsu(I3)

I4=cv2.imread('img4.png',0)
th4=threshold_otsu(I4)

print('Threshold of img3 is', th3)

print('Threshold of img4 is', th4)