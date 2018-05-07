import MLP_svhn
import MLPs_svhn
import SW_svhn
import tensorflow as tf

net = MLPs_svhn.Network(500)
net.run(100000)

tf.reset_default_graph()

net = MLPs_svhn.Network(1000)
net.run(100000)

tf.reset_default_graph()

net = MLPs_svhn.Network(2000)
net.run(100000)

tf.reset_default_graph()    #tf가 가지고있는 이전 망을 날려서, 다음 망 처리하도록 함.

net = SW_svhn.Network()
net.run(100000)

tf.reset_default_graph()

net = MLP_svhn.Network()
net.run(100000)
