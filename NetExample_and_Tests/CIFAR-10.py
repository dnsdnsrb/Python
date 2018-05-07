import tensorflow as tf
import numpy as np
import input_data
import os
import shutil
import Imagenet

batch_size = 64
test_size = batch_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X ,p_keep_conv, p_keep_hidden):

    #이렇게 표시하는게 훨씬 보기 낫다, placeholder를 쓰는 것만 아니면(run할 때 써야하므로) 여기 다 넣을 수 있다?
    w1 = init_weights([5, 5, 3, 32])        #(5,5), 필터 32개를 만든다.(필터들은 처음 가중치가 다르므로 모두 다른 값을 가진다)
    w2 = init_weights([5, 5, 32, 64])       #(5,5), 필터 64개를 만든다.
    w3 = init_weights([5, 5, 64, 128])      #(5,5), 필터 128개를 만든다.
    w4 = init_weights([128 * 8 * 8, 5000])   #128, 625 여기서 128*4*4이므로 reshape이 필요
    w5 = init_weights([5000, 1000])            #625, 10
    w6 = init_weights([1000, 19])
                                                                                            #(64, 64, 3)
    with tf.name_scope("conv1"):
        l1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides =[1, 1, 1, 1], padding='SAME'))         #(64, 64, 32)
    with tf.name_scope("maxpool2"):
        l2 = tf.nn.max_pool(l1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    #(32, 32, 32)
        l2 = tf.nn.dropout(l2, p_keep_conv)

    with tf.name_scope("conv3"):
        l3 = tf.nn.relu(tf.nn.conv2d(l2, w2, strides=[1, 1, 1, 1], padding='SAME'))         #(32, 32, 64)

    with tf.name_scope("maxpool4"):
        l4 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   #(16, 16, 64)
        l4 = tf.nn.dropout(l4, p_keep_conv)

    with tf.name_scope("conv5"):
        l5 = tf.nn.relu(tf.nn.conv2d(l4, w3, strides=[1, 1, 1, 1], padding='SAME'))         #(16, 16, 128)

    with tf.name_scope("maxpool6"):
        l6 = tf.nn.max_pool(l5, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')   #(8, 8, 128)
        l6 = tf.reshape(l6, [-1, w4.get_shape().as_list()[0]])
        l6 = tf.nn.dropout(l6, p_keep_conv)

    with tf.name_scope("fc7"):
        l7 = tf.nn.relu(tf.matmul(l6, w4))
        l7 = tf.nn.dropout(l7, p_keep_hidden)

    with tf.name_scope("fc8"):
        l8 = tf.nn.relu(tf.matmul(l7, w5))
        pyx = tf.nn.sigmoid(tf.matmul(l8, w6))


    return pyx

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#trX = trX.reshape(-1, 28, 28, 1)    #input img 28*28*1
#teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 64, 64, 3])
Y = tf.placeholder("float", [None, 19])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= py_x, labels= Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"): #accuracy는 tf를 이용하는게 좋다. tensorboard로 볼 수 있기 때문
    predict_op = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))
    tf.summary.scalar("accuracy", acc_op)


#네트워크 저장 관련
#saver = tf.train.Saver()
#global_step = tf.Variable(0, name='global_step', trainable=False)
#ckpt_dir = "./Networkfile/conv"
#if not os.path.exists(ckpt_dir):
#    os.makedirs(ckpt_dir)

#이미지넷
imagenet = Imagenet.Imagenet()
train_num, test_num = imagenet.getnum()


with tf.Session() as sess:
    #초기 설정
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    test_indices = np.arange(test_num)
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:batch_size]

    #그래프 관련
    graphPath = "./logs/conv_logs"
    if not os.path.exists(graphPath):
        os.makedirs(graphPath)
    shutil.rmtree(graphPath)
    writer = tf.summary.FileWriter(graphPath, sess.graph)


    merge = tf.summary.merge_all()

    #ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    #if ckpt and ckpt.model_checkpoint_path:
    #    print(ckpt.model_checkpoint_path)
    #    saver.restore(sess, ckpt.model_checkpoint_path)

    n = 0
    for i in range(10):
        print(i)
        for start, end in zip(range(0, train_num, batch_size), range(batch_size, train_num, batch_size)):

            #학습
            trX, trY = imagenet.getdata(start, end, train=True)
            trX = trX.reshape(-1, 64, 64, 3)

            sess.run(train_op, feed_dict={X: trX, Y: trY,
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

            #테스트
            teX, teY = imagenet.getdata(0, batch_size, train=False)
            teX = teX.reshape(-1, 64, 64, 3)

            output, acc = sess.run([py_x, acc_op], feed_dict={X: teX, Y: teY,
                                                                p_keep_conv: 1.0, p_keep_hidden: 1.0})
            print(output)
            n += 1
            #writer.add_summary(summary, n)
            print(n, acc)

        #global_step.assign(i).eval()    #그래프
        #saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step) #저장

#correct_prediction = tf.equal(tf.argmax)
