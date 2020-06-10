import tensorflow as tf
import time

from opt_einsum.backends import tensorflow
from tensorflow.keras import layers

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, layer_sizes, pow):
        super(CustomDenseLayer, self).__init__()
        self.weight = self.add_weight('kernel', shape = [layer_sizes[0], layer_sizes[1]])
        self.bias = tf.Variable(initial_value=tf.zeros(layer_sizes[1]), name='bias')
        self.pow = pow

    def call(self, input, training=False):
        weight = tf.pow(self.weight, self.pow)
        y = tf.matmul(input, weight) + self.bias

        return y

class CustomDenseLayerBlock(tf.keras.layers.Layer):
    def __init__(self, layer_sizes):
        super(CustomDenseLayerBlock, self).__init__()
        self.hidden_size = layer_sizes[0]
        self.output_size = layer_sizes[1]

    def build(self, input_shape):
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = CustomDenseLayer([input_shape[-1], self.hidden_size], 1)

        self.dense2 = CustomDenseLayer([input_shape[-1], self.hidden_size], 2)

        self.dense3 = CustomDenseLayer([input_shape[-1], self.hidden_size], 3)

        self.dense4 = CustomDenseLayer([input_shape[-1], self.hidden_size], 4)

        self.dense5 = tf.keras.layers.Dense(self.output_size)

    def call(self, inputs, training=False):
        inputs = self.flatten(inputs)
        outputs = inputs

        output1 = self.dense1(inputs)
        # outputs = self.bn1(outputs)
        # outputs = tf.nn.tanh(outputs)

        output2 = self.dense2(inputs)
        # outputs = self.bn2(outputs, training=False)
        # outputs = tf.nn.tanh(outputs)

        output3 = self.dense3(inputs)
        # outputs = self.bn3(outputs, training=False)
        # outputs = tf.nn.tanh(outputs)

        output4 = self.dense4(inputs)
        # outputs = self.bn4(outputs, training=False)
        # outputs = tf.nn.tanh(outputs)

        outputs = output1 + output2 + output3 + output4
        outputs = tf.nn.sigmoid(outputs)

        outputs = self.dense5(outputs)
        # outputs = self.bn5(outputs, training=False)
        # outputs = tf.nn.softmax(outputs)

        return outputs

class Network():
    def __init__(self):
        self.learning_rate = 1e-4
        self.model = self.make_model()
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def make_model(self):


        self.layer1 = tf.keras.layers.Flatten()
        self.layer2 = CustomDenseLayerBlock([250, 10])
        # self.layer2 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
        # self.layer3 = tf.keras.layers.Dense(250, activation=tf.nn.relu)
        # self.layer4 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

        model = tf.keras.Sequential([self.layer1, self.layer2])
        # model.add(tf.keras.layers.Dense(10))

        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(28 * 28))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dense(28 * 28))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dense(28 * 28))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dense(28 * 28))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dense(10))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Softmax())

        # model.add(CustomDenseLayerBlock([28*28, 28*28, 28*28, 28*28, 10]))

        return model

    def loss(self, label, pred):
        return self.cross_entropy(label, pred)

    def train_step(self, images, labels, train_var = None):
        with tf.GradientTape() as tape:
            pred = self.model(images, training=True)
            loss = self.loss(labels, pred)

        if train_var == None:
            train_var = self.model.trainable_variables

        gradients = tape.gradient(loss, train_var)
        self.optimizer.apply_gradients(zip(gradients, train_var))
        self.train_loss(loss)
        self.train_accuracy(labels, pred)

    def test_step(self, images, labels):
        pred = self.model(images, training=False)
        loss = self.loss(labels, pred)

        self.test_loss(loss)
        self.test_accuracy(labels, pred)


    def train(self, train_ds, test_ds, epochs):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        for epoch in range(epochs):
            start = time.time()
            for images, labels in train_ds:
                self.train_step(images, labels)
                # print(self.train_loss.result())

            # for images, labels in train_ds:
            #     self.train_step(images, labels, self.layer4.trainable_variables)
            #
            # for images, labels in train_ds:
            #     self.train_step(images, labels, self.layer3.trainable_variables)
            #
            # for images, labels in train_ds:
            #     self.train_step(images, labels, self.layer2.trainable_variables)

            for images, labels in test_ds:
                self.test_step(images, labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(time.time() - start, template.format(epoch+1,
                                                       self.train_loss.result(),
                                                       self.train_accuracy.result()*100,
                                                       self.test_loss.result(),
                                                       self.test_accuracy.result()*100))

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 10

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

net = Network()

net.train(train_ds, test_ds, EPOCHS)

