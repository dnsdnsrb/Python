import os
import tarfile
# import requests
import numpy as np
import random
import PIL
import tensorflow as tf

from PIL import ImageOps
from PIL import Image
# from nltk.corpus import wordnet
#from urllib.request import urlopen

#filepath = "../../imagenet_fall11_urls.tgz"



class Cifar():
    def __init__(self):
        self.dataset_path = "CIFAR/"
        self.trX = "trX.bin"
        self.trY = "trY.bin"
        self.teX = "teX.bin"
        self.teY = "teY.bin"
        self.image_bytes = 32 * 32 * 3
        self.label_bytes = 1
        self.onehot = 10
        self.record_bytes = 3073
        self.depth = 3
        self.height = 32
        self.width = 32

    def read_sets(self, filename_queue):
        class Record(object):
            pass

        records = Record()
        #train batch
        reader = tf.FixedLengthRecordReader(record_bytes=self.image_bytes + self.label_bytes)
        records.key, value = reader.read(filename_queue)    #key는 나오지만 쓰진않음, value만 사용

        record_bytes = tf.decode_raw(value, tf.uint8)
        records.label = tf.cast(tf.strided_slice(record_bytes, [0], [self.label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [self.label_bytes], [self.label_bytes + self.image_bytes]),
                [self.depth, self.height, self.width])
        # Convert from [depth, height, width] to [height, width, depth].
        records.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return records

    def create_batchs(self, images, labels, batch_size):
        images, labels = tf.train.batch([images, labels], batch_size=batch_size)

        images = tf.reshape(images, [batch_size, self.height, self.width, self.depth])
        labels = tf.reshape(labels, [batch_size, self.onehot])


        return images, labels

    def distorting(self, images):
        #images = tf.random_crop(images, [self.height * 0.75, self.width * 0.75, self.depth])  #무작위 크기 조절, [] 안의 값으로 크기 조정됨
        images = tf.image.random_flip_left_right(images)  #무작위 좌우반전
        images = tf.image.random_brightness(images, max_delta=63) #무작위 밝기 조절
        images = tf.image.random_contrast(images, lower=0.2, upper=1.8)   #무작위 대비 조절

        return images

    def create_sets(self, gray = False, distort = False, batch_size = 128): #test batch도 만들 것
        train_batches = [os.path.join(self.dataset_path, 'data_batch_%d.bin' % i) for i in range(1, 6)] #1~6 배치 파일 큐를 생성
        train_batches = tf.train.string_input_producer(train_batches)

        test_batch = [os.path.join(self.dataset_path, 'test_batch.bin')]
        test_batch = tf.train.string_input_producer(test_batch)


        #읽는 부분
        read_input = self.read_sets(train_batches)
        print(read_input.shape)
        trX = read_input.uint8image
        trY = read_input.label

        read_input = self.read_sets(test_batch)
        teX = read_input.uint8image
        teY = read_input.label

        # if distort == True:
        #     images = self.distorting(images)

        #형변환 부분
        trX = tf.cast(trX, tf.float32)                 #
        trX.set_shape([self.height, self.width, self.depth])
        trY.set_shape([1])                                             #
        trY = tf.one_hot(trY, 10, on_value=1, off_value=0)

        teX = tf.cast(teX, tf.float32)
        teX.set_shape([self.height, self.width, self.depth])
        teY.set_shape([1])  #
        teY = tf.one_hot(teY, 10, on_value=1, off_value=0)
        #print(read_input.label)


        #print(images, labels)

        #return images, labels
        #return 1

        trX, trY = self.create_batchs(trX, trY, batch_size)
        teX, teY = self.create_batchs(teX, teY, batch_size)

        return trX, trY, teX, teY


def Test():
    with tf.Session() as sess:
        Net = Cifar()
        images, labels = Net.create_sets()
        print(np.shape(images))
        print(sess.run(images[0]))

Test()