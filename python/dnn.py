import numpy
import struct
import cv2
import sys
import math
import csv
from scipy import signal

class new_network(object):
    def __init__(self):
        self.n=0.1
        self.u=0.5
        self.p=0.8
        self.neurons0=[]
        self.w01=[]
        self.b01=[]
        self.pre_dw01=[0]*6
        self.pre_db01=[0]*6
        for i in range(6):
            tempw=numpy.random.random(25).astype(numpy.float64)*2-1
            tempw.resize(5,5)
            self.w01.append(tempw)
            tempb=numpy.random.random(1).astype(numpy.float64)*2-1
            self.b01.append(tempb)
        self.neurons1=numpy.zeros((6,cols-4,rows-4),numpy.float64)
        self.neurons2=numpy.zeros((6,(cols-4)/2,(rows-4)/2),numpy.float64)
        self.w23=[]
        self.b23=[]
        self.pre_dw23=[[0]*6]*16
        self.pre_db23=[0]*16
        for i in range(16):
            tempws=[]
            for i in range(6):
                tempw=numpy.random.random(25).astype(numpy.float64)*2-1
                tempw.resize(5,5)
                tempws.append(tempw)
            self.w23.append(tempws)
            tempb=numpy.random.random(1).astype(numpy.float64)*2-1
            self.b23.append(tempb)
        self.neurons3=numpy.zeros((16,(cols-4)/2-4,(rows-4)/2-4),numpy.float64)
        self.neurons4=numpy.zeros((16,((cols-4)/2-4)/2,((rows-4)/2-4)/2),numpy.float64)
        self.w45=numpy.random.random(4*4*16*120).astype(numpy.float64)*2-1
        self.w45.resize(120,4*4*16)
        self.b45=numpy.random.random(120).astype(numpy.float64)*2-1
        self.b45.resize(120,1)
        self.pre_dw45=0
        self.pre_db45=0
        self.neurons5=[]
        self.dropout5=numpy.random.random(120)>self.p
        self.dropout5.resize(120,1)
        self.w56=numpy.random.random(120*84).astype(numpy.float64)*2-1
        self.w56.resize(84,120)
        self.b56=numpy.random.random(84).astype(numpy.float64)*2-1
        self.b56.resize(84,1)
        self.pre_dw56=0
        self.pre_db56=0
        self.neurons6=[]
        self.dropout6=numpy.random.random(84)>self.p
        self.dropout6.resize(84,1)
        self.w67=numpy.random.random(84*10).astype(numpy.float64)*2-1
        self.w67.resize(10,84)
        self.b67=numpy.random.random(10).astype(numpy.float64)*2-1
        self.b67.resize(10,1)
        self.pre_dw67=0
        self.pre_db67=0
        self.neurons7=[]
    def set_source(self,img):
        self.neurons0=img
    def change_dropout(self):
        self.dropout5=numpy.random.random(120)>self.p
        self.dropout5.resize(120,1)
        self.dropout6=numpy.random.random(84)>self.p
        self.dropout6.resize(84,1)
    def correction_dropout(self):
        self.w56=self.w56*self.p
        self.w67=self.w67*self.p
    def update(self):
        for i in range(6):
            self.neurons1[i,:,:]=self.sigmoid(signal.convolve2d(self.neurons0,self.w01[i],mode='valid')+self.b01[i])
        for i in range(6):
            self.neurons2[i,:,:]=cv2.resize(self.neurons1[i],(12,12))
        for i in range(16):
            temp=0
            for j in range(6):
                temp+=signal.convolve2d(self.neurons2[j],self.w23[i][j],mode='valid')
            self.neurons3[i,:,:]=self.sigmoid(temp+self.b23[i])
        for i in range(16):
            self.neurons4[i,:,:]=cv2.resize(self.neurons3[i],(4,4))
        self.neurons4.resize(256,1)
        self.neurons5=self.sigmoid(numpy.dot(self.w45,self.neurons4)+self.b45)
        self.neurons5[self.dropout5]=0.0
        self.neurons4.resize(16,4,4)
        self.neurons6=self.sigmoid(numpy.dot(self.w56,self.neurons5)+self.b56)
        self.neurons6[self.dropout6]=0.0
        self.neurons7=numpy.exp(numpy.dot(self.w67,self.neurons6)+self.b67)
        self.neurons7=self.neurons7/numpy.sum(self.neurons7)
    def calc_err(self,ans):
        self.delta7=self.neurons7-ans
    def adjust(self):
        self.delta6=numpy.dot(self.w67.transpose(),self.delta7)*self.dsigmoid(self.neurons6)
        self.delta6[self.dropout6]=0.0
        self.pre_dw67*=self.u
        self.pre_dw67+=-self.n*numpy.dot(self.delta7,self.neurons6.transpose())
        self.w67=self.w67+self.pre_dw67
        self.pre_db67*=self.u
        self.pre_db67+=-self.n*self.delta7
        self.b67=self.b67+self.pre_db67
        self.delta5=numpy.dot(self.w56.transpose(),self.delta6)*self.dsigmoid(self.neurons5)
        self.delta5[self.dropout5]=0.0
        self.pre_dw56*=self.u
        self.pre_dw56+=-self.n*numpy.dot(self.delta6,self.neurons5.transpose())
        self.w56=self.w56+self.pre_dw56
        self.pre_db56*=self.u
        self.pre_db56+=-self.n*self.delta6
        self.b56=self.b56+self.pre_db56
        self.neurons4.resize(256,1)
        self.delta4=numpy.dot(self.w45.transpose(),self.delta5)*self.dsigmoid(self.neurons4)
        self.pre_dw45*=self.u
        self.pre_dw45+=-self.n*numpy.dot(self.delta5,self.neurons4.transpose())
        self.w45=self.w45+self.pre_dw45
        self.pre_db45*=self.u
        self.pre_db45+=-self.n*self.delta5
        self.b45=self.b45+self.pre_db45
        self.delta4.resize(16,4,4)
        self.neurons4.resize(16,4,4)
        self.delta3=numpy.zeros((16,8,8),numpy.float64)
        for i in range(16):
            self.delta3[i,:,:]=cv2.resize(self.delta4[i],(8,8),interpolation=cv2.INTER_NEAREST)/4
        self.delta2=numpy.zeros((6,12,12),numpy.float64)
        for i in range(6):
            temp=0
            for j in range(16):
                temp+=signal.convolve2d(self.delta3[j],numpy.fliplr(numpy.flipud(self.w23[j][i])))
            self.delta2[i,:,:]=temp
        for i in range(16):
            for j in range(6):
                self.pre_dw23[i][j]*=self.u
                self.pre_dw23[i][j]+=-self.n*signal.convolve2d(numpy.fliplr(numpy.flipud(self.neurons2[j])),self.delta3[i],mode='valid')
                self.w23[i][j]=self.w23[i][j]+self.pre_dw23[i][j]
            self.pre_db23[i]*=self.u
            self.pre_db23[i]+=-self.n*numpy.sum(self.delta3[i])
            self.b23[i]=self.b23[i]+self.pre_db23[i]
        self.delta1=numpy.zeros((6,24,24),numpy.float64)
        for i in range(6):
            self.delta1[i,:,:]=cv2.resize(self.delta2[i],(24,24),interpolation=cv2.INTER_NEAREST)/4
        for i in range(6):
            self.pre_dw01[i]*=self.u
            self.pre_dw01[i]+=-self.n*signal.convolve2d(numpy.fliplr(numpy.flipud(self.neurons0)),self.delta1[i],mode='valid')
            self.w01[i]=self.w01[i]+self.pre_dw01[i]
            self.pre_db01[i]*=self.u
            self.pre_db01[i]+=-self.n*numpy.sum(self.delta1[i])
            self.b01[i]=self.b01[i]+self.pre_db01[i]
    def predict(self):
        max=0.0
        max_ind=-1
        for i in range(len(self.neurons7)):
            if self.neurons7[i][0]>max:
                max_ind=i
                max=self.neurons7[i][0]
        return max_ind
    def sigmoid(self,x):
        return 1/(1+numpy.exp(-x))
    def dsigmoid(self,x):
        return x*(1-x)
    def save(self,filew):
        csvout=csv.writer(filew)
        for i in range(6):
            csvout.writerows(self.w01[i])
            csvout.writerows([self.b01[i]])
        for i in range(16):
            for j in range(6):
                csvout.writerows(self.w23[i][j])
            csvout.writerows([self.b23[i]])
        csvout.writerows(self.w45)
        csvout.writerows(self.b45)
        csvout.writerows(self.w56)
        csvout.writerows(self.b56)
        csvout.writerows(self.w67)
        csvout.writerows(self.b67)
    def load(self,filer):
        cin=csv.reader(filer)
        data=[]
        for row in cin:
            new_row=[]
            for each in row:
                new_row.append(float(each))
            data.append(new_row)
        num=0
        for i in range(6):
            self.w01[i]=numpy.array(data[num:num+5],numpy.float64)
            num+=5
            self.b01[i]=numpy.array(data[num],numpy.float64)
            num+=1
        for i in range(16):
            for j in range(6):
                self.w23[i][j]=numpy.array(data[num:num+5],numpy.float64)
                num+=5
            self.b23[i]=numpy.array(data[num],numpy.float64)
            num+=1
        self.w45[:]=numpy.array(data[num:num+120],numpy.float64)
        num+=120
        self.b45[:]=numpy.array(data[num:num+120],numpy.float64)
        num+=120
        self.w56[:]=numpy.array(data[num:num+84],numpy.float64)
        num+=84
        self.b56[:]=numpy.array(data[num:num+84],numpy.float64)
        num+=84
        self.w67[:]=numpy.array(data[num:num+10],numpy.float64)
        num+=10
        self.b67[:]=numpy.array(data[num:num+10],numpy.float64)

#train
if __name__=='__main__':
    #file read
    filei=open('train-images.idx3-ubyte','rb')
    filel=open('train-labels.idx1-ubyte','rb')
    if (not filei) or (not filel):
        print("file not found")
        sys.exit()
    stri=filei.read(16)
    train_num=struct.unpack('>I',stri[4:8])[0]
    rows=struct.unpack('>I',stri[8:12])[0]
    cols=struct.unpack('>I',stri[8:12])[0]
    size=rows*cols
    forunpackbuf=str(size)+'b'
    strl=filel.read(8)
    if train_num!=struct.unpack('>I',strl[4:8])[0]:
        print("train_num does not match")
    imgs=[]
    anss=[]
    for i in range(train_num):
        
        stri=filei.read(size)
        strl=filel.read(1)
        img=numpy.array(struct.unpack(forunpackbuf,stri),numpy.uint8)
        img.resize((rows,cols))
        img=img.astype(numpy.float32)/255
        imgs.append(img)
        anss.append(struct.unpack('b',strl)[0])
    filei.close()
    filel.close()
    
    #construct neuralnet
    numpy.random.seed(0)
    net=new_network()
    correct=0
    err=10000
    pre_err=10000
    pre_pre_err=10000
    pre_pre_pre_err=10000
    for i in range(600000):#20 -> train_num
        #choose image
        j=numpy.random.randint(0,train_num)
        #add noise
        noise=numpy.random.randint(0,10)
        temp_img=imgs[j].copy()
        if noise!=0:
            temp_noise=numpy.random.random(size)
            temp_noise.resize(rows,cols)
            temp_noise_pos=temp_noise<=(noise/100.0)
            temp_img[temp_noise_pos]=temp_noise[temp_noise_pos]*100.0/noise

        #train
        net.set_source(temp_img)
        net.change_dropout()
        net.n=5000.0/(50000+i)
        net.update()
        ans=numpy.zeros((10,1))
        ans[anss[j]]=1
        net.calc_err(ans)
        net.adjust()
        
        #early stopping
        if net.predict()==anss[j]:
            correct+=1
        if i%10000==0:
            err=10000-correct
            correct=0
            if err>pre_err and pre_err>pre_pre_err and pre_pre_err>pre_pre_pre_err:
                net.correction_dropout()
                filew=open('network.csv','wt')
                net.save(filew)
                filew.close()
                print("early stoping")
                sys.exit()
            pre_pre_pre_err=pre_pre_err
            pre_pre_err=pre_err
            pre_err=err
        #for safe
        if i%10000==0:
            filew=open('network.csv','wt')
            net.save(filew)
            filew.close()
            
        #for debug
        if i%100==0:
            print net.neurons7
            print net.predict(),anss[j],i,err
    net.correction_dropout()
    filew=open('network.csv','wt')
    net.save(filew)
    filew.close()

#test
if __name__=='__main1__':
    #file read
    filei=open('t10k-images.idx3-ubyte','rb')
    filel=open('t10k-labels.idx1-ubyte','rb')
    if (not filei) or (not filel):
        print("file not found")
        sys.exit()
    stri=filei.read(16)
    test_num=struct.unpack('>I',stri[4:8])[0]
    rows=struct.unpack('>I',stri[8:12])[0]
    cols=struct.unpack('>I',stri[8:12])[0]
    size=rows*cols
    forunpackbuf=str(size)+'b'
    strl=filel.read(8)
    if test_num!=struct.unpack('>I',strl[4:8])[0]:
        print("test_num does not match")
    imgs=[]
    anss=[]
    for i in range(test_num):
        stri=filei.read(size)
        strl=filel.read(1)
        img=numpy.array(struct.unpack(forunpackbuf,stri),numpy.uint8)
        img.resize((rows,cols))
        img=img.astype(numpy.float32)/255
        imgs.append(img)
        anss.append(struct.unpack('b',strl)[0])
    filei.close()
    filel.close()
    
    #construct neuralnet
    net=new_network()

    filer=open('network.csv','rt')
    net.load(filer)    
    filer.close()

    net.p=1.0
    net.change_dropout()
    
    correct=0
    for i in range(test_num):
        net.set_source(imgs[i])
        net.update()
        if net.predict()==anss[i]:
            correct+=1
    print "precision:", float(correct)/test_num

    correct=0
    for noise in range(25):
        for i in range(test_num):
            temp_img=imgs[i].copy()
            temp_noise=numpy.random.random(size)
            temp_noise.resize(rows,cols)
            temp_noise_pos=temp_noise<=((noise+1)/100.0)
            temp_img[temp_noise_pos]=temp_noise[temp_noise_pos]*100.0/(noise+1)
            net.set_source(temp_img)
            net.update()
            if net.predict()==anss[i]:
                correct+=1
        print "precision noise",noise+1,"%", float(correct)/test_num
        correct=0

#visualize
if __name__=='__main2__':
    rows=28
    cols=28
    size=rows*cols
    #construct neuralnet
    net=new_network()
    
    filer=open('network.csv','rt')
    net.load(filer)    
    filer.close()
    
    omega01s=[]
    for i,omega in zip(range(6),net.w01):
        tempomega=cv2.flip(omega,-1)
        tempomega=(tempomega-numpy.min(tempomega))*255/(numpy.max(tempomega)-numpy.min(tempomega))
        omega01s.append(tempomega)
        tempomega=tempomega.astype(numpy.uint8)
        tempomega=cv2.resize(tempomega,(tempomega.shape[0]*10,tempomega.shape[1]*10),interpolation=cv2.INTER_NEAREST)
        cv2.namedWindow('image'+str(i))
        cv2.imshow('image'+str(i),tempomega)
        cv2.imwrite('w01_'+str(i)+'.jpg',tempomega)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    omega12s=[]
    for i in range(6):
        temp0=numpy.insert(omega01s[i],0,0,axis=0)
        temp1=numpy.insert(omega01s[i],5,0,axis=0)
        temp2=(numpy.insert(temp0,0,0,axis=1)+numpy.insert(temp0,5,0,axis=1)+numpy.insert(temp1,0,0,axis=1)+numpy.insert(temp1,5,0,axis=1))/4
        omega12s.append(temp2)

    omega23s=[]
    for i in range(16):
        temp=0
        for j in range(6):
            temp0=cv2.flip(net.w23[i][j],-1)
            for k in range(4):
                temp0=numpy.insert(temp0,4-k,0,axis=0)
            for k in range(4):
                temp0=numpy.insert(temp0,4-k,0,axis=1)
            temp=temp+signal.convolve2d(temp0,omega12s[j])
        temp=(temp-numpy.min(temp))*255/(numpy.max(temp)-numpy.min(temp))
        omega23s.append(temp)
        temp=temp.astype(numpy.uint8)
        temp=cv2.resize(temp,(temp.shape[0]*10,temp.shape[1]*10),interpolation=cv2.INTER_NEAREST)
        cv2.namedWindow('image'+str(i))
        cv2.imshow('image'+str(i),temp)
        cv2.imwrite('w23_'+str(i)+'.jpg',temp)
    cv2.waitKey(0)

    omega34s=[]
    for i in range(16):
        temp0=numpy.insert(omega23s[i],[0,0],[[0],[0]],axis=0)
        temp1=numpy.insert(omega23s[i],[14,14],[[0],[0]],axis=0)
        temp2=(numpy.insert(temp0,[0,0],[0,0],axis=1)+numpy.insert(temp0,[14,14],[0,0],axis=1)+numpy.insert(temp1,[0,0],[0,0],axis=1)+numpy.insert(temp1,[14,14],[0,0],axis=1))/4
        omega34s.append(temp2)

    omega45s=numpy.zeros((120,28,28),numpy.float64)
    for i in range(120):
        omega=net.w45[i,:].copy()
        omega.resize(16,4,4)
        temp=0
        for j in range(16):
            tempomega=omega[j,:,:].copy()
            for k in range(3):
                tempomega=numpy.insert(tempomega,[3-k,3-k,3-k],[[0],[0],[0]],axis=0)
            for k in range(3):
                tempomega=numpy.insert(tempomega,[3-k,3-k,3-k],[0,0,0],axis=1)
            temp=temp+signal.convolve2d(tempomega,omega34s[j])
        temp=(temp-numpy.min(temp))*255/(numpy.max(temp)-numpy.min(temp))
        omega45s[i,:,:]=temp
        temp=temp.astype(numpy.uint8)
        temp=cv2.resize(temp,(temp.shape[0]*10,temp.shape[1]*10),interpolation=cv2.INTER_NEAREST)
        #cv2.namedWindow('image'+str(i))
        #cv2.imshow('image'+str(i),temp)
        cv2.imwrite('w45_'+str(i)+'.jpg',temp)

    omega56s=numpy.dot(net.w56,omega45s.reshape(120,28*28))
    omega56s.resize(84,28,28)
    for i in range(84):
        omega56s[i,:,:]=(omega56s[i,:,:]-numpy.min(omega56s[i,:,:]))*255/(numpy.max(omega56s[i,:,:])-numpy.min(omega56s[i,:,:]))
        temp=omega56s[i,:,:].copy()
        temp=temp.astype(numpy.uint8)
        temp=cv2.resize(temp,(temp.shape[0]*10,temp.shape[1]*10),interpolation=cv2.INTER_NEAREST)
        print temp.shape
        cv2.imwrite('w56_'+str(i)+'.jpg',temp)

    omega67s=numpy.dot(net.w67,omega56s.reshape(84,28*28))
    omega67s.resize(10,28,28)
    for i in range(10):
        omega67s[i,:,:]=(omega67s[i,:,:]-numpy.min(omega67s[i,:,:]))*255/(numpy.max(omega67s[i,:,:])-numpy.min(omega67s[i,:,:]))
        temp=omega67s[i,:,:].copy()
        temp=temp.astype(numpy.uint8)
        temp=cv2.resize(temp,(temp.shape[0]*10,temp.shape[1]*10),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('w67_'+str(i)+'.jpg',temp)
        
    temp=net.w67.copy()
    temp0=numpy.zeros((temp.shape[0],temp.shape[1],3),numpy.uint8)
    temp=temp*255/(max(numpy.max(temp),-numpy.min(temp)))
    #temp=temp*255/numpy.maximum(numpy.max(temp,axis=1,keepdims=True),-numpy.min(temp,axis=1,keepdims=True))
    tempp=temp.copy()
    tempp[tempp<0]=0.0
    tempm=temp.copy()
    tempm[tempm>0]=0.0
    tempm=tempm*-1
    temp0[:,:,2]=tempp.astype(numpy.uint8)
    temp0[:,:,0]=tempm.astype(numpy.uint8)
    temp0=cv2.resize(temp0,(840,100),interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow('image67')
    cv2.imshow('image67',temp0)
    cv2.imwrite('w67.jpg',temp0)
    cv2.imwrite('w67.png',temp0)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

#test parallel
if __name__=='__main3__':
    #file read
    filei=open('t10k-images.idx3-ubyte','rb')
    filel=open('t10k-labels.idx1-ubyte','rb')
    if (not filei) or (not filel):
        print("file not found")
        sys.exit()
    stri=filei.read(16)
    test_num=struct.unpack('>I',stri[4:8])[0]
    rows=struct.unpack('>I',stri[8:12])[0]
    cols=struct.unpack('>I',stri[8:12])[0]
    size=rows*cols
    forunpackbuf=str(size)+'b'
    strl=filel.read(8)
    if test_num!=struct.unpack('>I',strl[4:8])[0]:
        print("test_num does not match")
    imgs=[]
    anss=[]
    for i in range(test_num):
        stri=filei.read(size)
        strl=filel.read(1)
        img=numpy.array(struct.unpack(forunpackbuf,stri),numpy.uint8)
        img.resize((rows,cols))
        img=img.astype(numpy.float32)/255
        imgs.append(img)
        anss.append(struct.unpack('b',strl)[0])
    filei.close()
    filel.close()
    
    #construct neuralnet
    net1=new_network()
    net2=new_network()
    net3=new_network()
    
    filer=open('network1.csv','rt')
    net1.load(filer)    
    filer.close()

    filer=open('network2.csv','rt')
    net2.load(filer)    
    filer.close()

    filer=open('network3.csv','rt')
    net3.load(filer)    
    filer.close()
  
    net1.p=1.0
    net1.change_dropout()

    net2.p=1.0
    net2.change_dropout()

    net3.p=1.0
    net3.change_dropout()

    correct=0
    for i in range(test_num):
        net1.set_source(imgs[i])
        net1.update()
        net2.set_source(imgs[i])
        net2.update()
        net3.set_source(imgs[i])
        net3.update()

        tempans=numpy.zeros((10,1),numpy.float64)
        tempans+=net1.neurons7
        tempans+=net2.neurons7
        tempans+=net3.neurons7
        max_num=0.0
        max_ind=-1
        for j in range(len(tempans)):
            if tempans[j][0]>max_num:
                max_ind=j
                max_num=tempans[j][0]
        if max_ind==anss[i]:
            correct+=1
        if i%100==0:
            print tempans
            print max_ind, anss[i], i

    print "precision:", float(correct)/test_num

    correct=0
    for noise in range(25):
        for i in range(test_num):
            temp_img=imgs[i].copy()
            temp_noise=numpy.random.random(size)
            temp_noise.resize(rows,cols)
            temp_noise_pos=temp_noise<=((noise+1)/100.0)
            temp_img[temp_noise_pos]=temp_noise[temp_noise_pos]*100.0/(noise+1)

            net1.set_source(temp_img)
            net1.update()
            net2.set_source(temp_img)
            net2.update()
            net3.set_source(temp_img)
            net3.update()
            
            tempans=numpy.zeros((10,1),numpy.float64)
            tempans+=net1.neurons7
            tempans+=net2.neurons7
            tempans+=net3.neurons7
            max_num=0.0
            max_ind=-1
            for j in range(len(tempans)):
                if tempans[j][0]>max_num:
                    max_ind=j
                    max_num=tempans[j][0]
            if max_ind==anss[i]:
                correct+=1
                        
        print "precision noise",noise+1,"%", float(correct)/test_num
        correct=0
