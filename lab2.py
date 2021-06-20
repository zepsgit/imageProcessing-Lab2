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
k4=gkern(7,1.2)
#convolve the mask over the image
#padding zeros 
def padding(kx):
    global kernel_n
    kernel_n=kx.shape[0]
    global pn
    pn=(kernel_n-1)/2 #padding number
    pn=int(pn)
    global img_pad
    img_pad=np.zeros([m+kernel_n-1,n+kernel_n-1])
    for i in range(pn,m+pn):
        for j in range(pn,n+pn):
                img_pad[i,j]=img[i-pn,j-pn]
    return img_pad


def filter(kx):
    kernel_n=kx.shape[0]
    img_new=np.zeros([m,n])
    sum_p=0
    img_pad=padding(kx)
    for s in range(0,m):
        for t in range(0,n):
            for p in range(0,kernel_n):
               for q in range(0,kernel_n):
                    sum_p=sum_p+img_pad[s+p,t+q]*kx[p,q]
                    img_new[s,t]=sum_p
                    sum_p=0
    return img_new

img_ones3=filter(k1)
img_ones3 = img_ones3.astype(int)
cv2.imwrite('ones3.png', img_ones3)

img_ones7=filter(k2)
img_ones7 = img_ones7.astype(int)
cv2.imwrite('ones7.png', img_ones7)

img_g3=filter(k3)
img_g3=img_g3.astype(int)
cv2.imwrite('img_g3.png',img_g3)

img_g7=filter(k4)
img_g7=img_g7.astype(int)
cv2.imwrite('img_g7.png',img_g7)

cv2.imwrite('origin.png',img)
cv2.imshow('origin',img)
cv2.imshow('ones3',img_ones3)

cv2.imshow('ones7',img_ones7)
cv2.imshow('img_g3',img_g3)
cv2.imshow('img_g7',img_g7)
cv2.waitKey(0)
cv2.destroyAllWindows()