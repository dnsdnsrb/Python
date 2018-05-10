import tensorflow as tf
import numpy as np
import svhn
import cifar
import mnist
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import flags

import NNutils
from datetime import datetime

class Dataset:
    def __init__(self, name, x_size, y_size, train_data, train_label, test_data, test_label):
        self.name = name
        self.x_size = x_size
        self.y_size = y_size
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

def select_dataset(name):
    x_size, y_size, train_data, train_label, test_data, test_label = 0, 0, [], [] ,[] ,[] #초기화
    if name=='cifar':
        dataset = cifar.CIFAR()
        train_data, train_label, test_data, test_label = dataset.getdata()

        train_data = train_data.reshape(-1, 3072)
        test_data = test_data.reshape(-1, 3072)
        x_size = 3072
        y_size = 10

    elif name=='svhn':
        dataset = svhn.SVHN()
        train_data, train_label = dataset.get_trainset()
        test_data, test_label = dataset.get_testset()

        train_data = train_data.reshape(-1, 3072)
        test_data = test_data.reshape(-1, 3072)
        x_size = 3072
        y_size = 10

    elif name=='mnist':
        dataset = mnist.read_data_sets(flags.MNIST_DIR, one_hot=True)
        train_data, train_label, test_data, test_label = dataset.train.images, dataset.train.labels, \
                                                         dataset.test.images, dataset.test.labels
        x_size = 784
        y_size = 10

    elif name=='news':
        trainset = fetch_20newsgroups(data_home=flags.NEWS_DIR, subset='train')
        testset = fetch_20newsgroups(data_home=flags.NEWS_DIR, subset='test')

        vectorizer = TfidfVectorizer(analyzer='word', max_features=3072)

        vectorizer.fit(trainset.data)
        train_data = vectorizer.transform(trainset.data)
        train_data = csr_matrix.todense(train_data)
        train_label = trainset.target
        train_label = NNutils.onehot(train_label, 20, list=True)
        # print(train_label.shape)

        test_data = vectorizer.transform(testset.data)
        test_data = csr_matrix.todense(test_data)
        test_label = testset.target
        test_label = NNutils.onehot(test_label, 20, list=True)

        x_size = 3072
        y_size = 20

    return Dataset(name, x_size, y_size, train_data, train_label, test_data, test_label)


class Network():
    def __init__(self, x_size, y_size, layers, regularization=False, batch_normalization=False, droprate=.0):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.y_size = y_size

        self.x = tf.placeholder("float32", [None, x_size])
        self.y = tf.placeholder("float32", [None, y_size])
        self.keeprate = tf.placeholder("float32")
        self.training = tf.placeholder("bool")

        self.droprate = droprate

        self.layers = layers

        self.train_list = []

        self.activation = "relu"
        self.network = "mlp"

        if regularization == True:
            self.regularize = 0.001
            self.regularizer = tf.contrib.layers.l2_regularizer(self.regularize)
            self.network += "_regul"
        else:
            self.regularizer = None

        if batch_normalization == True:
            self.network += "_bn"
            self.bn = True
        else:
            self.bn = False

        self.train()



    def fc_layer(self, x, output_num, activation, dropout):
        # if activation == 'relu':
        #     activation = tf.nn.relu
        # elif activation == 'sigmoid':
        #     activation = tf.nn.sigmoid
        # elif activation == 'tanh':
        #     activation = tf.nn.tanh
        # else:
        #     activation == None

        output = tf.layers.dense(x, output_num, kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=self.regularizer)
        if self.bn == True:
            output = tf.layers.batch_normalization(output, training=self.training)
        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        elif activation == 'tanh':
            output = tf.nn.tanh(output)
        # output = tf.layers.dropout(output, rate=dropout, training=self.training)
        output = tf.nn.dropout(output, 1 - dropout)

        y = output
        return y

    def info(self):
        layers_sum = str(sum(self.layers))
        layers_num = str(len(self.layers))
        dropout = str(self.droprate)
        return self.network + "-" + self.activation + "-" + layers_num + "-" + layers_sum + "-d" + dropout

    def model(self, x): #[3072, 450, 300, 200, 100, 50, 10]
        layers = self.layers
        image_size = 32
        output = x
        reshape_size = 0

        layer_num = 0

        for layer in layers:
            layer_num += 1
            with tf.variable_scope('fc' + str(layer_num)):
                output = self.fc_layer(output, layer, self.activation, self.droprate)

        with tf.variable_scope('fc'):
            output = tf.contrib.layers.fully_connected(output, self.y_size, activation_fn=None)

        y = output
        return y

    def train(self):

        #learning rate
        with tf.name_scope("learning_rate"):
            learning_rate = tf.train.exponential_decay(0.001,
                                                       self.global_step,
                                                       (50000 / self.batch_size) * 10,
                                                       0.95, staircase=True)
            learning_rate = tf.maximum(0.0001, learning_rate)
            tf.summary.scalar("learning_rate", learning_rate)

        #model
        y_ = self.model(self.x)

        #loss and training
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))

            #정규항 추가
            if self.regularizer != None:
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                self.loss += reg_term

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                    minimize(self.loss, global_step=self.global_step)

            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, dataset, step_limit):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()


            name = self.info()
            path = dataset.name + "/" + str(step_limit) + name

            # path = "mlp/" + str(step_limit) + "SvhnMlpRelu" + str(len(self.layers))
            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)
                for start, end in zip(range(0, len(dataset.train_data), self.batch_size),
                                      range(self.batch_size, len(dataset.train_data), self.batch_size)):
                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.opt, self.loss,
                                     self.global_step],
                                    feed_dict={self.x: dataset.train_data[start:end],
                                               self.y: dataset.train_label[start:end],
                                               self.keeprate: 1 - self.droprate,
                                               self.training: True})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, \
                loss, \
                accuracy = sess.run([merged, self.loss, self.accuracy],
                                    feed_dict={self.x: dataset.test_data,
                                               self.y: dataset.test_label,
                                               self.keeprate: 1.0,
                                               self.training: False})

                writer_test.add_summary(summary, step)
                print("test results : ", accuracy, loss)
                saver.save(sess, path + "/" + name + ".ckpt", step)

if __name__ == "__main__":
    # dataset = select_dataset('mnist')
    # model = Network(dataset.x_size, dataset.y_size, [525])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('mnist')
    # model = Network(dataset.x_size, dataset.y_size, [1050])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()

    # dataset = select_dataset('svhn')
    # model = Network(dataset.x_size, dataset.y_size, [750])
    # model.run(dataset, 100000+0*895)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('svhn')
    # model = Network(dataset.x_size, dataset.y_size, [1500])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()

    # dataset = select_dataset('cifar')
    # model = Network(dataset.x_size, dataset.y_size, [750])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('cifar')
    # model = Network(dataset.x_size, dataset.y_size, [1500])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [750])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [1500])
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()

    # combine test = mlp + bn + dropout

    dataset = select_dataset('mnist')
    model = Network(dataset.x_size, dataset.y_size, [400, 200, 100],
                    batch_normalization=True,
                    droprate=0.5)
    model.run(dataset, 100000)

    tf.reset_default_graph()

    dataset = select_dataset('svhn')
    model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100],
                    batch_normalization=True,
                    droprate=0.5)
    model.run(dataset, 100000)

    tf.reset_default_graph()

    dataset = select_dataset('cifar')
    model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100],
                    batch_normalization=True,
                    droprate=0.5)
    model.run(dataset, 100000)

    tf.reset_default_graph()

    dataset = select_dataset('news')
    model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100],
                    batch_normalization=True,
                    droprate=0.5)
    model.run(dataset, 100000)

    tf.reset_default_graph()

    # dataset = select_dataset('svhn')
    # model = Network(dataset.x_size, dataset.y_size, [750], droprate=0.0)
    # model.run(dataset, 100000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('svhn')
    # model = Network(dataset.x_size, dataset.y_size, [1500], droprate=0.0)
    # model.run(dataset, 100000)
    # #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100], droprate=0.5)
    # model.run(dataset, 50000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100], droprate=0.75)
    # model.run(dataset, 50000)
    #
    # tf.reset_default_graph()
    #
    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100], droprate=0.25)
    # model.run(dataset, 50000)

    # tf.reset_default_graph()

    # dataset = select_dataset('news')
    # model = Network(dataset.x_size, dataset.y_size, [600, 450, 300, 200, 100], droprate=.99)
    # model.run(dataset, 30000)
    #
    # tf.reset_default_graph()
