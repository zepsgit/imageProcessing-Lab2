import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#read image
img = cv2.imread('img1.png',0)
#obtain number of rows and columns 
#of the imaage

m,n=img.shape
#develop different filter
k1=np.ones([3,3],dtype=int)
k1=k1/9
k2=np.ones([7,7],dtype=int)
k2=k2/49
#creates gaussian kernel with side length l and a sigma of sig
def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)
k3=gkern(3,0.5)
k4=gkern(12,1.2)
#convolve the mask over the image
def filter(kx):
    kernel_n=kx.shape[0]
    kn=int((kernel_n-1)/2)
    img_=np.zeros([m,n])
    img_new=np.zeros([m+kernel_n,n+kernel_n])
    for s in range(kn,m+kn):
        for t in range(kn,n+kn):
            img_new[s,t]=img[s-kn,t-kn]
            sum_p=0
            for p in range(0,kernel_n):
               for q in range(0,kernel_n):
                    sum_p=sum_p+img_new[s-kn+p,t-kn+q]*kx[p,q]
                    img_[s-kn,t-kn]=sum_p
    return img_

img_f=filter(k1)
img_f = img_f.astype(np.uint8)
print(img_f.min(), img_f.max())
cv2.imwrite('ones3.png', img_f)
cv2.imshow('ones3',img_f)

cv2.imwrite('origin.png',img)
cv2.imshow('origin',img)

cv2.waitKey(0)
cv2.destroyAllWindows()