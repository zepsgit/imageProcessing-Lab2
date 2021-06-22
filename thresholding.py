import cv2
import numpy as np

def findTh(fig):
    img=cv2.imread(fig,0)
    global L
    L=img.max()
    m, n=img.shape
        #cnt is the number of intensity, pt is the probability accordingly
    cnt=np.zeros(L+1)
    pt=np.zeros(L+1)
    for pixel in img:
        cnt[pixel]+=1
        pt[pixel]=cnt[pixel]/(m*n)

    def sigma(t): 
        wt0=wt1=ut0=ut1=0
        for x in range(0,t):
            wt0=wt0+pt[x]
        for y in range(t,L+1):
            wt1=wt1+pt[y]
        for p in range(0,t):
            ut0=ut0+x*pt[p]/wt0
        for q in range(t,L+1):
            ut1=ut1+q*pt[q]/wt1
        sigmaSqr=wt0*wt1*(ut0-ut1)**2
        return sigmaSqr

    maxInit=0
    for k in range(0,L+1):
        s=sigma(k)
        if s>maxInit:
            maxInit=s
            th=k
    return th
#binarize the iamge using threshold found by above function
def biImag(fig,th):
    img=cv2.imread(fig,0)
    L=img.max()
    m,n=img.shape
    for x in range(0,m):
        for y in range(0,n):
            if img[x,y]<th:
                img[x,y]=0
            else:
                img[x,y]=L
    return img

th3=findTh('img3.png')
print('Threshold of img3 is', th3)
th4=findTh('img4.png')
print('Threshold of img4 is', th4)

I3=biImag('img3.png',th3)
I4=biImag('img4.png',th4)
cv2.imshow('img3',I3)
cv2.imshow('img4',I4)
cv2.waitKey(0)
cv2.destroyAllWindows
cv2.imwrite('img3bi.png',I3)
cv2.imwrite('img4bi.png',I4)


    
    
    
    
