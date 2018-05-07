import tensorflow as tf
import numpy as np
import svhn
import NNutils
import flags
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def fc_layer(name, x, output_num, pow, activation='sigmoid', dropout=None):
    weight = tf.get_variable(name, [x.get_shape().as_list()[1], output_num],
                             initializer=tf.random_normal_initializer(stddev=0.01))
    bias = tf.Variable(tf.zeros(output_num))

    weight = tf.pow(weight, pow)
    y = tf.matmul(x, weight) + bias

    if activation == 'relu':
        y = tf.nn.relu(y)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
    elif activation == 'tanh':
        y = tf.nn.tanh(y)
    elif activation == 'none':
        pass
    else:
        print("unknown")

    if not dropout is None:
        y = tf.nn.dropout(y, dropout)

    return y

class Network():
    def __init__(self, layer=150):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.layer = layer

        self.x = tf.placeholder("float32", [None, 3072])
        self.y = tf.placeholder("float32", [None, 20])

        self.dropout_conv = tf.placeholder("float")
        self.dropout_normal = tf.placeholder("float")

        self.train_list = []

        self.activation = "tanh"
        self.network = "rw"

    def info(self):
        layer = str(self.layer)
        exponent = 0    #지수, 아직 이 변수를 이용하게 짜진 않음
        return self.network + "-" + self.activation + "-" + layer


    def model(self, x):
        image_size = 32
        output = x
        reshape_size = 0

        layer = self.layer
        layer_num = 0

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output1 = fc_layer('fc', output, layer, 1, activation=self.activation, dropout=self.dropout_normal)

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output2 = fc_layer('fc', output, layer, 2, activation=self.activation, dropout=self.dropout_normal)

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output3 = fc_layer('fc', output, layer, 3, activation=self.activation, dropout=self.dropout_normal)

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output4 = fc_layer('fc', output, layer, 4, activation=self.activation, dropout=self.dropout_normal)

        layer_num += 1
        with tf.variable_scope('fc' + str(layer_num)):
            output5 = fc_layer('fc', output, layer, 5, activation=self.activation, dropout=self.dropout_normal)

        with tf.variable_scope('sum' + str(layer_num)):
            output = output1 + output2 + output3 + output4 + output5
            # output = tf.nn.sigmoid(output)    #이것도 info에 넣긴해야하는데?

        with tf.variable_scope('fc'):
            output = tf.contrib.layers.fully_connected(output, 20, activation_fn=None)

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

        #cost and training
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.cost, global_step=self.global_step)

            tf.summary.scalar("cost", self.cost)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, step_limit):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # dataset = fetch_20newsgroups(data_home="C:/Projects/Data/20newsgroups", subset='all')
            trainset = fetch_20newsgroups(data_home=flags.NEWS_DIR, subset='train')
            testset = fetch_20newsgroups(data_home=flags.NEWS_DIR, subset='test')

            vectorizer = TfidfVectorizer(analyzer='word', max_features=3072)

            vectorizer.fit(trainset.data)
            train_data = vectorizer.transform(trainset.data)
            train_data = csr_matrix.todense(train_data)
            train_label = trainset.target
            train_label = NNutils.onehot(20, train_label)
            # print(train_label.shape)

            test_data = vectorizer.transform(testset.data)
            test_data = csr_matrix.todense(test_data)
            test_label = testset.target
            test_label = NNutils.onehot(20, test_label)

            # test_indices = np.arange(len(test_data))
            # np.random.shuffle(test_indices)
            # test_indices = test_indices[0:1000]

            name = self.info()
            path = "news/" + str(step_limit) + name

            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)
                for start, end in zip(range(0, len(train_data), self.batch_size),
                                      range(self.batch_size, len(train_data), self.batch_size)):
                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.training, self.cost,
                                     self.global_step],
                                    feed_dict={self.x: train_data[start:end],
                                               self.y: train_label[start:end],
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 1.0})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, \
                loss, \
                accuracy = sess.run([merged, self.cost, self.accuracy],
                                    feed_dict={self.x: test_data,
                                               self.y: test_label,
                                               self.dropout_conv: 1.0,
                                               self.dropout_normal: 1.0})

                writer_test.add_summary(summary, step)
                print("test results : ", accuracy, loss)
                saver.save(sess, path + "/" + name + ".ckpt", step)

if __name__ == "__main__":
    model = Network()
    model.run(100000)