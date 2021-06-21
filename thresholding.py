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

a=findTh('img3.png')
print(a)


    
    
    
    
