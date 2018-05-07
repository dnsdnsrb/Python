import numpy as np
from scipy.io import loadmat


class Utils:
    def onehot(self, length, origin):  # (onehot 배열 크기, 1이 되야하는 인덱스)
        onehot = np.zeros([len(origin), length], dtype=np.uint8)
        onehot[range(len(origin)), origin] = 1

        return onehot

    def convert_images2arr(self, images, pixel_depth=255.0):  #pixel_depth는 픽셀이 가지는 값 크기(ex> rgb의 r값의 범위)
        rows = images.shape[0]
        cols = images.shape[1]
        channels = images.shape[2]  #depth라고 부르기도 함
        num_images = images.shape[3]

        scalar = 1.0 / pixel_depth  #사용 안하려면 pixel_depth=1.0으로 하면 된다.

        new_array = np.empty(shape=(num_images, rows, cols, channels), dtype=np.float32)

        for x in range(0, num_images):  #normalize 부분, 0~1값으로 변환한다.
            channels = images[:, :, :, x]
            norm_vector = (255-channels) * scalar
            new_array[x] = norm_vector

        return new_array

class SVHN(Utils):
    def __init__(self):
        self.dataset_path = "../Data/SVHN"
        self.trainset = "../Data/SVHN/train_32x32.mat"
        self.testset = "../Data/SVHN/test_32x32.mat"

    def process_data(self, file):
        data = loadmat(file)
        images = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0
        labels_onehot = self.onehot(10, labels)
        image_array = self.convert_images2arr(images, pixel_depth=1.0)

        return  image_array, labels_onehot

    def get_trainset(self):
        with open(self.trainset, 'rb') as file:
            data = self.process_data(file)

        return data

    def get_testset(self):
        with open(self.testset, 'rb') as file:
            data = self.process_data(file)

        return data

if __name__ == "__main__":
    dataset = SVHN()

    train_images, train_labels = dataset.get_trainset()
    test_images, test_labels = dataset.get_testset()
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    print(train_images[0].shape)
    print(train_images[0], train_labels[0])