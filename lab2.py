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
    img_pad=np.zeros([m+2*pn,n+2*pn])
    for i in range(pn,m+pn):
        for j in range(pn,n+pn):
                img_pad[i,j]=img[i-pn,j-pn]
    return img_pad


def filter(kx):
    kernel_n=kx.shape[0]
    img_new=np.zeros([m+kernel_n,n+kernel_n])
    sum_p=0
    img_pad=padding(kx)
    for ki in range(pn,pn+m):
        for kj in range(pn,pn+n):
            for x in range(ki,ki+kernel_n):
                for y in range(kj,kj+kernel_n):
                    sum_p=sum_p+img_pad[x-pn,y-pn]*kx[x-ki,y-kj]
                    img_new[x,y]=sum_p
    return img_new
                       
img_f=filter(k1)
img_f = img_f.astype(np.uint8)
#cv2.imwrite('blurred.tif', img_f)
cv2.imshow("blurred image",img_f)
cv2.waitKey(0)
cv2.destroyAllWindows()