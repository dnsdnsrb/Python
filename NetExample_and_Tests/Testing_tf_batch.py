import tensorflow as tf
import numpy as np
import NNutils
import DataSet

from datetime import datetime

dataset = DataSet.Cifar()
trX, trY, teX, teY = dataset.create_sets()


def NN(X):
    w1 = tf.Variable(tf.random_normal([32*32*3, 2048], stddev=0.1))
    b1 = tf.Variable(tf.zeros([2048]))

    w2 = tf.Variable(tf.random_normal([2048, 1400], stddev=0.1)) #MNIST 숫자가 10개이므로(0~9) 10이다.
    b2 = tf.Variable(tf.zeros([1400]))

    w3 = tf.Variable(tf.random_normal([1400, 900], stddev=0.1))
    b3 = tf.Variable(tf.zeros([900]))

    w4 = tf.Variable(tf.random_normal([900, 600], stddev=0.1))
    b4 = tf.Variable(tf.zeros([600]))

    w5 = tf.Variable(tf.random_normal([600, 300], stddev=0.1))
    b5 = tf.Variable(tf.zeros([300]))

    w6 = tf.Variable(tf.random_normal([300, 10], stddev=0.1))
    b6 = tf.Variable(tf.zeros([10]))

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
    Y = tf.matmul(l5, w6) + b6

    return Y

#tf.contrib.framework.get_or_create_global_step(graph=None)
global_step = tf.Variable(0, trainable=False, name="global_step")

with tf.name_scope("cost"):
    Y_ = NN(trX)  #tf를 이용한 배치는 placeholder를 사용하지 않는다. => 바로 이미지를 집어넣음.

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Y_, labels= trY))   #마찬가지
    tf.add_to_collection('costs', cost)
    total_cost = tf.add_n(tf.get_collection('costs'), 'total_cost')

    cost_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    costs = tf.get_collection('costs')
    cost_averages_op = cost_averages.apply(costs + [total_cost])
    tf.summary.scalar("cost", cost)

with tf.control_dependencies([cost_averages_op]):
    train_op = tf.train.AdamOptimizer(0.001).minimize(total_cost, global_step)


with tf.name_scope("accuracy"):
    Y_test = NN(teX)

    predict_op = tf.equal(tf.arg_max(teY, 1), tf.arg_max(Y_test, 1))
    acc_op = tf.reduce_mean(tf.cast(predict_op, "float"))
    tf.summary.scalar("accuracy", acc_op)

 #tf를 이용한 배치는 placeholder를 사용하지 않는다. => 바로 이미지를 집어넣음.
# Y_test = NN(teX)
# cost_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= Y_test, labels= teY))   #마찬가지
# top_k_op = tf.nn.in_top_k(Y_test, tf.argmax(teY, 1), 1)





#데이터 관련

#이 형태는 문제점이 train, test 함수를 따로 구현해야한다. => feeddict가 안돼서 그럼.
with tf.train.MonitoredTrainingSession(checkpoint_dir="Networkfile/tftest", #세이브, 그래프, 초기화 모두 다 포함되있음.
                                       hooks=[tf.train.StopAtStepHook(last_step=1000000)],
                                       save_checkpoint_secs=1000000,
                                       save_summaries_steps=50) as monsess:
    #writer = tf.summary.FileWriter(monsess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    while not(monsess.should_stop()):
        step, loss, _ = monsess.run([global_step ,cost, train_op], options=run_options, run_metadata=run_metadata) #feed_dict는 tensor는 들어갈 수 없다.

        #tf.add_run_metadata()
        if step % 50 == 0:
            accuracy, loss = monsess.run([acc_op, cost])
            print(step, accuracy, loss)

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     sess.run(train_op)


    #accuracy, loss = sess.run([acc_op, cost], feed_dict={X: teX, Y: teY})
    #print(datetime.now(), i, accuracy, loss)

