import cv2
import numpy as np

img=cv2.imread('img1.png',0)
m,n=img.shape

def medf(size):
    #listK=np.zeros([size,size])
    listK=np.zeros([size,size],dtype=np.uint8)
    kn=int((size-1)/2)
    img_=np.zeros([m,n],dtype=np.uint8)
    img_new=np.zeros([m+size-1,n+size-1],dtype=np.uint8)
    for s in range(kn,kn+m):
        for t in range(kn,kn+n):
            img_new[s,t]=img[s-kn,t-kn]
            for p in range(0,size):
                for q in range(0,size):
                    listK[p,q]=img_new[s-kn+p,t-kn+q]
                    listKsort=np.sort(listK, axis=None, kind='quicksort')
                    med=listKsort[int((size*size-1)/2)]
                    img_[s-kn,t-kn]=med
    return img_

m3=medf(3)
m3=m3.astype(np.uint8)
cv2.imwrite('m3.png',m3)
cv2.imshow('m3',m3)

m5=medf(5)
m5=m5.astype(np.uint8)
cv2.imwrite('m5.png',m5)
cv2.imshow('m5',m5)

cv2.waitKey(0)
cv2.destroyAllWindows