import numpy as np 
import random
np.random.seed(612)
arr1=np.random.rand(1,1000)
#print(arr1[0,0])
usernum=int(input("请输入一个1~100之间的整数："))
i=0
while i<1000:
    if(i%usernum==0):
        print(arr1[0,i])
    i+=1