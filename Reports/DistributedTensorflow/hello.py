import tensorflow as tf

c = tf.constant("Hell, distributed Tensorflow")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
print(sess.run(c))