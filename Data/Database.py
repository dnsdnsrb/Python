import os
import tarfile
import requests
import numpy as np
import random
import PIL
import tensorflow as tf

from PIL import ImageOps
from PIL import Image
from nltk.corpus import wordnet
#from urllib.request import urlopen

class Cifar:
    def __init__(self):
        self.dataset_path = "CIFAR/"
        self.trX = "trX.bin"
        self.trY = "trY.bin"
        self.teX = "teX.bin"
        self.teY = "teY.bin"
        self.image_size = 32 * 32 * 3
        self.label_size = 10

        s

    def getnum(self):
        sizeY = os.path.getsize(os.path.join(self.dataset_path, self.trY))
        sizeX = os.path.getsize(os.path.join(self.dataset_path, self.trX))
        numX = (sizeX / self.image_size)
        numY = (sizeY / self.label_size) #/ datasize

        test_size = os.path.getsize(os.path.join(self.dataset_path, self.teY))
        test_num = test_size / self.label_size #/ datasize

        if not numX == numY:
            print("X, Y 데이터 수 불일치", numX, numY)
            return -1

    def gettest(self):
        class DataSets:
            pass

        datas = DataSets()

        trX = open(os.path.join(self.dataset_path, self.trX), "rb")
        trY = open(os.path.join(self.dataset_path, self.trY), "rb")
        teX = open(os.path.join(self.dataset_path, self.teX), "rb")
        teY = open(os.path.join(self.dataset_path, self.teY), "rb")

        train_num = 50000
        test_num = 10000

        train_images = self.read(trX, self.image_size * train_num, self.image_size)  # * data_size
        train_labels = self.read(trY, self.label_size * train_num, self.label_size)  # * data_size
        test_images = self.read(teX, self.image_size * test_num, self.image_size)
        test_labels = self.read(teY, self.image_size * test_num, self.label_size)

        train_images = train_images.reshape(-1, 32, 32, 3)
        train_images = train_images.astype('float32')

        train_labels = train_labels.reshape(-1, 10)
        train_labels = train_labels.astype('float32')

        test_images = test_images.reshape(-1, 32, 32, 3)
        test_images = test_images.astype('float32')

        test_labels = test_labels.reshape(-1, 10)
        test_labels = test_labels.astype('float32')

        trX.close()
        trY.close()
        teX.close()
        teY.close()

        return train_images, train_labels, test_images, test_labels

    def create_sets(self):
        # 출력값을 뽑아내는 과정을 거치면서, 동시에 배치는 하나로 뭉친 새로운 파일을 만들자
        #batch = open(os.path.join(self.cifar_path, "data_batch_1.bin"), "rb") #배치에 X, Y가 뭉쳐있음
        trX = open(os.path.join(DataSet.dataset_path, "trX.bin"), "wb")
        trY = open(os.path.join(DataSet.dataset_path, "trY.bin"), "wb")

        teX = open(os.path.join(DataSet.dataset_path, "teX.bin"), "wb")
        teY = open(os.path.join(DataSet.dataset_path, "teY.bin"), "wb")

        test = open(os.path.join(DataSet.dataset_path, "test_batch.bin"), "rb")
        data_size = DataSet.image_size + 1
        file_size = test.seek(0, os.SEEK_END)
        test.seek(0)

        #train batch
        for i in range(1, 6):
            with open(os.path.join(DataSet.dataset_path, "data_batch_" + str(i) + ".bin"), "rb") as batch:

                data = DataSet.read(self, batch, file_size, data_size, 'uint8')

                X = data[:,1:]
                X = np.reshape(X, (-1, 3, 32, 32))
                X = np.transpose(X, (0, 2, 3, 1))

                Y = data[:,0]  #출력값 뽑아냄    3073에서 첫 1byte가 출력, onehot으로 바꿔야함?
                Y = self.onehot(10, Y)

                trX.write(bytes(X))
                trY.write(bytes(Y))

        #test batch
        data = DataSet.read(self, test, file_size, data_size, 'uint8')

        X = data[:, 1:]
        X = np.reshape(X, (-1, 3, 32, 32))
        X = np.transpose(X, (0, 2, 3, 1))

        Y = data[:, 0]
        Y = self.onehot(10, Y)

        teX.write(bytes(X))
        teY.write(bytes(Y))

        trX.close()
        trY.close()
        teX.close()
        teY.close()
        test.close()

    def read(self, file, file_size, data_size, dtype='uint8'):
        data = file.read(file_size)
        data = np.frombuffer(data, dtype=dtype)
        data = np.reshape(data, (-1, data_size))

        return data

    def onehot(self, length, origin):   #(onehot 배열 크기, 1이 되야하는 인덱스)

        onehot = np.zeros([len(origin), length], dtype=np.uint8)
        onehot[range(len(origin)), origin] = 1
        #print(np.shape(onehot))

        return onehot

def test():
    Net = Cifar()
    #Net.create_sets()
    im, la, tim, tla = Net.gettest()
    im = im.astype('uint8')
    print(np.shape(im))

    image = Image.fromarray(im[128*80])
    print(la[128*80])
    image.show()


    #Net.getnum()
    #X, Y = Net.getdata(1, 1, train=True)
    #X = X.reshape(-1, 32, 32, 3)
    #print(Y)
    #arr = X[0]
    #arr = arr.astype('uint8')
    #im = Image.fromarray(arr)
    #im.show()

test()




#Net = Imagenet()
#Net.cifar_getdata()
#Net.getnum()
#Net.url_extract(20000, 40000)
#Net.create_sets()
#print(Net.getnum())
#listX, listY = Net.getdata(1, 5, train=True)
#Net.cifar_create()
#listX, listY = Net.CIFAR(0, 100, train=True)
#print(np.shape(listY))

#Net = Cifar()

#Net.create_sets()
#print(Net.getnum())
#X, Y = Net.getdata(0, 3845, train=True)

#print( X.dtype, Y.dtype)
#print(np.shape(X))
#print(Y[0])
#print(Y[3840])
#print(np.shape(Y))
#X.reshape([5, 32, 32, 3])
#X = tf.transpose(X, [0, 3, 1, 2])

#print(np.shape(X))

#im = Image.fromarray(X[0])
#im.show()
#print(X[0][3000])

#print(np.shape(listX), listX[0], listY[0])
#Net.show_wordlist()
#word = Net.wordnet()
#print(word[5787])
#list = Net.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)
#print(list)
#x, y = Net.getdata(8000, 8001, train=True)
#print(np.shape(y))
#print(y[0])
#url에서 이미지 저장
"""
def ImageSave(id, url, imagepath):
    with Image.open(requests.get(url, stream=True).raw) as im:
        print(id)
        try:
            im.save(imagepath)
        except(OSError):#가끔 이상하게 오류 떠서 이렇게 해봄, RGB가 아닌 경우 저장하다 오류나서 그런 듯
            os.remove(imagepath)
            im.convert('RGB').save(imagepath)
#

def test(self):
    filelist = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))] #os.listdir = 해당 경로의 모든 파일과 경로 출력, 이 줄 전체는 그 리스트에서 파일만 뽑아내는 것(폴더 제외)
    #filelist = os.listdir(filepath)
    print(filelist)




#DistributedNet(Canceled) 이미지 저장 파일 경로 생성
imagenetPath = "DistributedNet(Canceled)"
if not os.path.exists(imagenetPath):
    os.mkdir(imagenetPath)
#


#img = ImageOps.fit(img, size, Image.ANTIALIAS) #이미지 크기 조정
print("Started")
with open("fall11_urls.txt", "r", encoding="utf-8") as txt:
    for _ in range(8148):
        txt.readline()
    for i in range(8148, 10000):
        print(i)
        id, url = txt.readline().split()

        imagepath = os.path.join(imagenetPath, id + '.jpg')
        if not os.path.exists(imagepath):  #해당 이미지가 없다면 링크를 열어 저장하려 한다. exists 대신 isfile 써도 됨.
            try:
                ImageSave(id, url, imagepath)
            except (OSError): #이미지 링크가 깨지거나 한 경우 표시로서 빈 폴더를 만들고 통과한다.
                #if not os.path.exists(imagepath):
                    os.mkdir(imagepath)
        else:
            print(id, "is already exist.")
"""
#list = Net.wordnet()
#print(list[7846])

