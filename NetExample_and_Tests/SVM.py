import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import input_data
# Global variables.
BATCH_SIZE = 128  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of training epochs.')
tf.app.flags.DEFINE_float('svmC', 1,
                          'The C parameter of the SVM cost function.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')
FLAGS = tf.app.flags.FLAGS


# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
#   The given file should be a comma-separated-values (CSV) file saved by the savetxt command.
def main(argv=None):
    # Extract it into numpy matrices.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_labels, test_data, test_labels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # Convert labels to +1,-1
    train_labels[train_labels == 0] = -1

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the C param of SVM
    svmC = 0.5

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, 28*28])
    y = tf.placeholder("float", shape=[None, 10])

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([28*28, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_raw = tf.matmul(x, W) + b

    # Optimization.
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 10]),
                                          1 - y * y_raw))
    svm_loss = regularization_loss + svmC * hinge_loss
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    # Evaluation.
    predicted_class = tf.sign(y_raw)
    correct_prediction = tf.equal(y, predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as session:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()

        for i in range(100):
            for start, end in zip(range(0, len(train_data), 128), range(128, len(train_data), 128)):
                _, cost = session.run([train_step, svm_loss],
                                      feed_dict={x:train_data[start:end], y:train_labels[start:end]})
            print(cost)

if __name__ == '__main__':
    tf.app.run()