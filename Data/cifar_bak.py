import os
import tarfile
import numpy as np
import random
from PIL import Image
from scipy.io import loadmat

"""
이미지 변형(밝기 조절 같은 것)도 해서 데이터 양을 늘려서 AE와 NN을 비교해보자
CAE와 CNN 비교도 해보자
데이터에는 잘못된 데이터도 껴있음(깨진 링크따위)
이미지 이름에는 워드넷 사전 번호가 있으므로 그걸 지도학습

-데이터 만드는 법-
이미지 이름을 모두 확인 => wordnet 번호를 뽑아냄 => list 수만큼 출력 갯수가 된다.
이미지 크기는 모두 동일하게 변경
처음 나온 단어의 이미지는 test set에 넣음(처음 나온 단어를 기억하고 다른 단어가 나올 때까지 train set에 넣는다)

"""
class DataSet:
    def __init__(self):
        self.dataset_path = ""  #자식 클래스에서 할당됨.
        self.trX = "trX.bin"
        self.trY = "trY.bin"
        self.teX = "teX.bin"
        self.teY = "teY.bin"
        self.image_size = 0
        self.label_size = 0
        self.dtype = 0

    def extract(self, path): #tar 압축 해제
        tarfile.open(path, 'r').extractall()

    def onehot(self, length, origin):   #(onehot 배열 크기, 1이 되야하는 인덱스)

        onehot = np.zeros([len(origin), length], dtype=np.uint8)
        onehot[range(len(origin)), origin] = 1
        #print(np.shape(onehot))

        return onehot

    def sample(self, list, sample_length):   #random sample을 실행
        return random.sample(list, sample_length)

    def getdata(self, onehot = True):   #데이터 배치를 받을 때 사용하는 함수, 현재는 cifar만 됨.
        trX = open(os.path.join(self.dataset_path, self.trX), "rb")
        trY = open(os.path.join(self.dataset_path, self.trY), "rb")
        teX = open(os.path.join(self.dataset_path, self.teX), "rb")
        teY = open(os.path.join(self.dataset_path, self.teY), "rb")

        train_num = 50000
        test_num = 10000

        train_images = self.read(trX, self.image_size * train_num , self.image_size) #* data_size
        train_labels = self.read(trY, self.label_size * train_num , self.label_size)   #* data_size
        test_images = self.read(teX, self.image_size * test_num , self.image_size)
        test_labels = self.read(teY, self.image_size * test_num , self.label_size)

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

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def getgray(self):
        trX = open(os.path.join(self.dataset_path, self.trX), "rb")
        teX = open(os.path.join(self.dataset_path, self.teX), "rb")

        train_num = 50000
        test_num = 10000

        train_images = self.read(trX, self.image_size * train_num, self.image_size)  # * data_size
        test_images = self.read(teX, self.image_size * test_num, self.image_size)

        train_images = train_images.reshape(-1, 32, 32, 3)
        train_images = np.dot(train_images[..., :3], [0.299, 0.587, 0.114])
        train_images = train_images.astype('float32')

        test_images = test_images.reshape(-1, 32, 32, 3)
        test_images = np.dot(test_images[..., :3], [0.299, 0.587, 0.114])
        test_images = test_images.astype('float32')

        trX.close()
        teX.close()

        return train_images, test_images

    def getnum(self, datasize = 4):
        sizeY = os.path.getsize(os.path.join(self.dataset_path, self.trY))
        sizeX = os.path.getsize(os.path.join(self.dataset_path, self.trX))
        numX = (sizeX / self.image_size)
        numY = (sizeY / self.label_size) #/ datasize

        test_size = os.path.getsize(os.path.join(self.dataset_path, self.teY))
        test_num = test_size / self.label_size #/ datasize

        if not numX == numY:
            print("X, Y 데이터 수 불일치", numX, numY)
            return -1

        return int(numX), int(test_num)

    def read(self, file, file_size, data_size, dtype='uint8'):
        data = file.read(file_size)
        data = np.frombuffer(data, dtype=dtype)
        data = np.reshape(data, (-1, data_size))

        return data

class Cifar(DataSet):
    def __init__(self):
        DataSet.dataset_path = "../Data/CIFAR"
        DataSet.trX = "trX.bin"
        DataSet.trY = "trY.bin"
        DataSet.teX = "teX.bin"
        DataSet.teY = "teY.bin"
        DataSet.image_size = 32 * 32 * 3
        DataSet.label_size = 10
        DataSet.dtype = np.uint8

    def create_sets(self):  #기존 데이터 배치를 수정하여 새 걸 만듬.
        trX = open(os.path.join(DataSet.dataset_path, "trX.bin"), "wb")
        trY = open(os.path.join(DataSet.dataset_path, "trY.bin"), "wb")

        teX = open(os.path.join(DataSet.dataset_path, "teX.bin"), "wb")
        teY = open(os.path.join(DataSet.dataset_path, "teY.bin"), "wb")

        test = open(os.path.join(DataSet.dataset_path, "test_batch.bin"), "rb")
        data_size = DataSet.image_size + 1
        file_size = test.seek(0, os.SEEK_END)
        test.seek(0)

        #train batch
        for i in range(1, 6):   #원래의 데이터 배치는 라벨과 이미지가 같이 붙어있는데, 이 것을 따로 때냄.
            with open(os.path.join(DataSet.dataset_path, "data_batch_" + str(i) + ".bin"), "rb") as batch:

                data = DataSet.read(self, batch, file_size, data_size, 'uint8')

                X = data[:,1:]
                X = np.reshape(X, (-1, 3, 32, 32))
                X = np.transpose(X, (0, 2, 3, 1))

                Y = data[:,0]  #출력값 뽑아냄    3073에서 첫 1byte가 출력, onehot으로 바꿈.(onehot을 안 쓸거면 변경 필요)
                Y = self.onehot(10, Y)

                trX.write(bytes(X))
                trY.write(bytes(Y))

        #test batch 위와 동일
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

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def test(): #걍 테스트용
    Net = Cifar()
    Net.create_sets()
    im, la, tim, tla = Net.getdata()
    #
    # print(np.shape(im))
    im = im.astype('uint8')

    path = "image"
    create_path(path)
    # arr_uint = arr_uint.reshape(-1, 32 * 32 * 1)
    for i in range(128):
        image_path = path + "/" + str(i) + ".jpg"

        image = Image.fromarray(im[i])
        image.save(image_path)