import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        # 텐서플로우 클래스 입력 인자 상속 및 입력 인자 선언
        super(CustomDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # 나머지 필요한 선언
        self.kernel = self.add_variable("kernel", shape = [int(input_shape[-1]),
                                                           self.num_outputs])

    def call(self, input):
        # 실제 동작(연산)
        return tf.matmul(input, self.kernel)

class ResnetIdenetityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdenetityBlock, self).__init__(name='')
        filters1, filter2, filter3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, 1)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filter2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filter3, 1)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdenetityBlock(1, [1, 2, 3])


layer = CustomDenseLayer(10)
print("yeah")