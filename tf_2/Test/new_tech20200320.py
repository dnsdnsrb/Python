import tensorflow as tf
import time
from tensorflow.keras import layers

class CustomDenseLayerBlock(tf.keras.layers.Layer):
    def __init__(self, layer_sizes):
        super(CustomDenseLayerBlock, self).__init__()

        self.flatten = tf.keras.layers.Flatten()

        self.dense = [tf.keras.layers.Dense(layer_sizes[i]) for i in range(len(layer_sizes))]
        self.bn = [tf.keras.layers.BatchNormalization() for _ in range(len(layer_sizes))]

        self.dense_last = tf.keras.layers.Dense(10)
        self.bn_last = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        inputs = self.flatten(inputs)
        outputs = [0 for _ in range(len(self.dense))]

        for i in range(len(self.dense)):
            output = self.dense[i](inputs)
            output = self.bn[i](output, training=False)
            outputs[i] = tf.nn.tanh(output)

        output = tf.math.add_n(outputs)


        output = self.dense_last(output)
        output = self.bn_last(output)
        output = tf.nn.softmax(output)

        # outputs = self.dense5(outputs)
        # outputs = self.bn5(outputs, training=False)
        # output = tf.nn.softmax(outputs)

        return output

class Network():
    def __init__(self):
        self.learning_rate = 1e-4
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.layer_n = 7
        self.model = self.make_model()
        self.optimizers = [tf.keras.optimizers.Adam(self.learning_rate * (3*i + 1)) for i in range(self.layer_n)]

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def make_model(self):

        model = tf.keras.Sequential()
        model.add(CustomDenseLayerBlock([28*28 for _ in range(self.layer_n - 1)]))

        return model

    def loss(self, label, pred):
        return self.cross_entropy(label, pred)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            pred = self.model(images, training=True)
            loss = self.loss(labels, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        for i in range(len(self.optimizers)):
            self.optimizers[i].apply_gradients(zip(gradients[i*4:i*4+4], self.model.trainable_variables[i*4:i*4+4]))

        # self.optimizer1.apply_gradients(zip(gradients[:4], self.model.trainable_variables[:4]))
        # self.optimizer2.apply_gradients(zip(gradients[4:8], self.model.trainable_variables[4:8]))
        # self.optimizer3.apply_gradients(zip(gradients[8:12], self.model.trainable_variables[8:12]))
        # self.optimizer4.apply_gradients(zip(gradients[12:16], self.model.trainable_variables[12:16]))
        # self.optimizer5.apply_gradients(zip(gradients[16:20], self.model.trainable_variables[16:20]))
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

