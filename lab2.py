import cv2
import numpy as np

#gaussian kernel generator
def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

#kernel generator
'''
type='average'|'gaussian'|'sobel_h'|'sobel_v'|'laplacian'|'median'
size=odd number like 3,5,7...
sigma >=0
'''
def knl_generator(type, size, sigma):
    if type=='average':
        return np.ones([size,size],dtype=int)/(size*size)
    if type=='gaussian':
        return gkern(size, sigma)
    if type=='sobel_v':
        return np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    if type=='sobel_h':
        return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    if type=='laplacian':
        return np.array([[0.4038,    0.8021,    0.4038],
                        [0.8021,   -4.8233,    0.8021],
                        [0.4038,    0.8021,    0.4038]])
    if type=='median':
        return np.zeros([size,size])

#pad image border with 0
#size denote kernel size
def pad_img(img,size):
    m,n=img.shape
    pad_len=int((size-1)/2)#padding length for each edge
    padded_img=np.zeros([m+size-1,n+size-1])
    for i in range(pad_len, pad_len+m):
        for j in range(pad_len, pad_len+n):
            padded_img[i,j]=img[i-pad_len,j-pad_len]
    return padded_img

'''
function: spatial filter including linear and non-linear filter
input:img--np.array|type--'linear' or 'nonlinear'--median filter
output:filtered image--np.array
'''
def filter(img, type, kx):
    m,n=img.shape
    size=kx.shape[0]
    pad_len=int((size-1)/2)
    new_img=np.zeros([m,n],dtype=np.uint8)
    padded_img=pad_img(img, size)
    for i in range(pad_len,m+pad_len):
        for j in range(pad_len,n+pad_len):
            sum_prod=0
            for s in range(0,size):
               for t in range(0,size):
                   p=i-pad_len+s
                   q=j-pad_len+t
                   if type=='linear':
                       sum_prod=sum_prod+padded_img[p,q]*kx[s,t]
                   if type=='nonlinear':
                       kx[s,t]=padded_img[p, q]
            if type=='linear':
                new_img[i-pad_len,j-pad_len]=sum_prod
            if type=='nonlinear':
                sorted_kx=np.sort(kx,axis=None,kind='quicksort')
                med=sorted_kx[int((size*size-1)/2)]
                new_img[i-pad_len, j-pad_len]=med
    return new_img

#linear filter
img1=cv2.imread('img1.png',0)
img2=cv2.imread('img2.png',0)
k1=knl_generator('average', 3, 0)
img_one3=filter(img1, 'linear', k1)
cv2.imwrite('ones3.png', img_one3)
cv2.imshow('ones3',img_one3)
cv2.imwrite('ones3.png',img_one3)

k2=knl_generator('average', 7, 0)
img_one7=filter(img1,'linear',k2)
cv2.imwrite('ones7.png', img_one7)
cv2.imshow('ones7',img_one7)
cv2.imwrite('ones7.png',img_one7)

k3=knl_generator('gaussian', 3, 0.5)
img_g3=filter(img1,'linear',k3)
cv2.imshow('g3',img_g3)
cv2.imwrite('g3.png',img_g3)

k4=knl_generator('gaussian', 7, 1.2)
img_g7=filter(img1,'linear',k4)
cv2.imshow('g7',img_g7)
cv2.imwrite('g7.png',img_g7)

k6=knl_generator('sobel_h', 0, 0)
img_k6=filter(img2,'linear',k6)
cv2.imshow('k6',img_k6)
cv2.imwrite('k6.png',img_k6)

k7=knl_generator('sobel_v', 0, 0)
img_k7=filter(img2,'linear',k7)
cv2.imshow('k7',img_k7)
cv2.imwrite('k7.png',img_k7)

k8=knl_generator('laplacian', 0, 0)
img_k8=filter(img2,'linear',k8)
print(img_k8.max(), img_k8.min())
cv2.imshow('k8',img_k8)
cv2.imwrite('k8.png',img_k8)

#perform median filter
m3=knl_generator('median', 3, 0)
img_m3=filter(img1, 'nonlinear', m3)
cv2.imwrite('m3.png',img_m3)
cv2.imshow('m3',img_m3)

m5=knl_generator('median', 5, 0)
img_m5=filter(img1, 'nonlinear', m5)
cv2.imwrite('m5.png',img_m5)
cv2.imshow('m5',img_m5)

cv2.imwrite('origin1.png',img1)
cv2.imshow('originImg1',img1)

cv2.imwrite('origin2.png',img2)
cv2.imshow('originImg2',img2)

cv2.waitKey(0)
cv2.destroyAllWindows()