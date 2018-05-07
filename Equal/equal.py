import tensorflow as tf
import numpy as np
import sys
sys.path.append('../Data/')
import Imagenet
import NNutils

#
# train_num = np.arange(len(train_data))
# np.random.shuffle(train_num)
# train_num = train_num.reshape(-1, 2)
# first = train_num[:, 0]
# second = train_num[:, 1]





# count = 0
# for label_num in range(10):
#     for i in range(len(train_label)):
#         if train_label[0] == train_label[i]:
#             count += 1
#
#         train_data[0:128],
# print(count)



#train_label[train_num[0][0]]
#
# for i in range(4):
#     for i_num in range(i):
#         print(i, i_num)

def de_onehot(label):
    label = np.argmax(label, 1)
    return label

class Network():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder("float32", [None, 32, 32, 3])
        self.y = tf.placeholder("float32", [None, 10])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

    # def model(self, layers, x): #layers = [[name, output_num] ... [name, output_num]], x = input
    #     output = x
    #     after_conv = False
    #     image_width = 32
    #     for layer_num in range(len(layers)):
    #         name = layers[layer_num][0] + str(layer_num)
    #         print(layers[layer_num][0], after_conv)
    #
    #         if layers[layer_num][0] == 'normal' and after_conv == True:
    #             for i_num in range(layer_num):
    #                 image_width = int(image_width / 2)
    #
    #             with tf.name_scope("reshape"):  #layer num은 증가시키지 않음 => 실제로 layers에 포함되지 않고 알아서 만들어준다.
    #                 output = tf.reshape(output, [-1, image_width * image_width * layers[layer_num][1]])
    #
    #         with tf.variable_scope(name):
    #             if not layer_num == len(layers):
    #                 output = NNutils.create_layer(layers[layer_num][0], output, layers[layer_num][1],
    #                                               kernel_shape=[4, 4])
    #             else:
    #                 output = NNutils.create_layer(layers[layer_num][0], output, layers[layer_num][1],
    #                                               kernel_shape=[4, 4], activation='none')
    #
    #
    #         if layers[layer_num][0] == 'conv':
    #             after_conv = True
    #
    #             maxpool = "maxpool" + str(layer_num)
    #
    #             with tf.name_scope(maxpool):
    #                 output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #         else:
    #             after_conv = False
    #
    #
    #
    #     y = output
    #     return y

    def model(self, x, layer):
        image_width = 32

        output = x

        #first image
        layer_num = 0
        output1 = output

        with tf.variable_scope('1conv' + str(layer_num)):
            output1 = NNutils.create_layer('conv', output1, layer[layer_num])
            output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('1conv' + str(layer_num)):
            output1 = NNutils.create_layer('conv', output1, layer[layer_num])
            output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('1conv' + str(layer_num)):
            output1 = NNutils.create_layer('conv', output1, layer[layer_num])
            output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('1conv' + str(layer_num)):
            output1 = NNutils.create_layer('conv', output1, layer[layer_num])
            output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.name_scope('reshape'):
            output1 = tf.reshape(output1, [-1, image_width * image_width * layer[3]])

        #second image
        layer_num = 0
        output2 = output

        with tf.variable_scope('2conv' + str(layer_num)):
            output2 = NNutils.create_layer('conv', output2, layer[layer_num])
            output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('2conv' + str(layer_num)):
            output2 = NNutils.create_layer('conv', output2, layer[layer_num])
            output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('2conv' + str(layer_num)):
            output2 = NNutils.create_layer('conv', output2, layer[layer_num])
            output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.variable_scope('2conv' + str(layer_num)):
            output2 = NNutils.create_layer('conv', output2, layer[layer_num])
            output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            layer_num += 1

        with tf.name_scope('reshape'):
            output2 = tf.reshape(output2, [-1, image_width * image_width * layer[3]])

        #summation layer
        with tf.name_scope('concat'):
            output = tf.concat([output1, output2], 0)

        with tf.variable_scope('normal' + str(layer_num)):
            output = NNutils.create_layer('normal', output, layer[layer_num])
            layer_num += 1

        with tf.variable_scope('normal' + str(layer_num)):
            output = NNutils.create_layer('normal', output, layer[layer_num], activation='none')
            layer_num += 1

        y = output

        return y

    def train(self):

        #learning rate
        learning_rate = tf.train.exponential_decay(0.001,
                                                   self.global_step,
                                                   (50000 / self.batch_size) * 10,
                                                   0.95, staircase=True)
        learning_rate = tf.maximum(0.00001, learning_rate)

        #model
        y_ = self.model()

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

    def run(self):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            dataset = Imagenet.Cifar()
            train_data, train_label, test_data, test_label = dataset.getdata()
            train_label = de_onehot(train_label)
            test_label = de_onehot(test_label)

            path = "Equal/32-64-128-256-375-08-05"

            #어떻게 2개를 구성할까? => 랜덤은 50000개 중 2500개만 맞게됨, 일정구성은 5000개, 20000개는 맞게 하고 싶다.



# x = tf.placeholder("float32", [None, 32, 32, 3])
# y = tf.placeholder("float32", [None, 10])
# model = Network()
# # model.model([['conv', 32], ['conv', 64], ['conv', 128], ['conv', 256], ['normal', 375], ['normal', 10]], x)
# model.model(x, [32, 64, 128, 256, 375, 10])
#print('conv' + str(1))
dataset = Imagenet.Cifar()
train_data, train_label, test_data, test_label = dataset.getdata()
train_label = de_onehot(train_label)
test_label = de_onehot(test_label)

half = int(len(train_data) / 2)

first = train_data[:half]
second = train_data[half:]
print(len(first), len(second))

count = 0
for i in range(len(first)):
    if train_label[i] == train_label[half + i]:
        count += 1
print(count)