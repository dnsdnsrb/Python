import tensorflow as tf
from tensorflow.keras import layers

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input1, input2):
        super(CustomDenseLayer, self).__init__()
        self.input1 = input1
        self.input2 = input2
        self.total_input = input1 + input2
        self.dense = tf.keras.layers.Dense(self.total_input)

    def call(self, input):
        return self.dense(input)

class CustomDenseLayerBlock(tf.keras.layers.Layer):
    def __init__(self, layer_sizes):
        super(CustomDenseLayerBlock, self).__init__()
        layer1, layer2, layer3, layer4, layer5 = layer_sizes

        self.dense1 = tf.keras.layers.Dense(layer1)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.dense2 = tf.keras.layers.Dense(layer2)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.dense3 = tf.keras.layers.Dense(layer3)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dense4 = tf.keras.layers.Dense(layer4)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.dense5 = tf.keras.layers.Dense(layer5)
        self.bn5 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        outputs = self.dense1(inputs)
        outputs = self.bn1(outputs)
        outputs = tf.nn.tanh(outputs)

        outputs = self.dense2(inputs + outputs)
        outputs = self.bn2(outputs, training=False)
        outputs = tf.nn.tanh(outputs)

        outputs = self.dense3(inputs + outputs)
        outputs = self.bn3(outputs, training=False)
        outputs = tf.nn.tanh(outputs)

        outputs = self.dense4(inputs + outputs)
        outputs = self.bn4(outputs, training=False)
        outputs = tf.nn.tanh(outputs)

        outputs = self.dense5(inputs + outputs)
        outputs = self.bn5(outputs, training=False)
        outputs = tf.nn.tanh(outputs)

        return outputs

class Network():
    def __init__(self):
        self.learning_rate = 1e-4
        pass

    def generate_model(self):

        model = tf.keras.Sequential()

        model.add(CustomDenseLayerBlock([28*28, 14*14, 8*8, 4*4, 10]))

        return model

    def generate_loss(self):
        model = self.generate_model()
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)




    def train(self):

        pass
