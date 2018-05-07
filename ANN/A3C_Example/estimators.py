import tensorflow as tf

def build_shared_network(x, add_summaries=False):
    conv1 = tf.layers.conv2d(x, 16, 8, 4, activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, 32, 4, 2, activation=tf.nn.relu, name="conv2")

    fc1 = tf.layers.dense(tf.layers.flatten(conv2), 256, name="fc1")

    if add_summaries:
        tf.contrib.layers.summarize_activation(conv1)
        tf.contrib.layers.summarize_activation(conv2)
        tf.contrib.layers.summarize_activation(fc1)

    return fc1

class PolicyEstimator():
    def __init__(self, num_ouptuts, reuse=False, trainable=True):
        self.num_outputs = num_ouptuts

        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        x = tf.to_float(self.states) / 255.0
        batch_size = tf.shape(self.states)[0]

        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(x, add_summaries=(not reuse))

        with tf.variable_scope("policy_net"):
            self.logits = tf.layers.dense(fc1, num_ouptuts, activation=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8

            self.predictions = {"logits": self.logits, "probs": self.probs}

            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            # 배열을 리스트처럼 만듬 => 각 데이터의 시작 부분(offset) + action값(onehot 아님) = action의 위치
            # 그 후  tf.gather을 이용해 원하는 action에 해당하는 확률값만 뽑아냄
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01*self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram(self.entropy.op.name, self.entropy)

            if trainable:
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                # grad가 None인 경우 학습이 망가지는 것을 막기 위해서 이렇게 만든 듯 하다.
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                # 여기 train_op 정작 쓰진 않음. worker에서 apply_gradient를 함. 지워도 될 듯
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=tf.train.get_global_step())

                var_scope_name = tf.get_variable_scope().name
                summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
                summaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
                summaries = [s for s in summary_ops if var_scope_name in s.name]
                self.summaries = tf.summary.merge(summaries)

class ValueEstimator():
    def __init__(self, reuse=False, trainable=True):

        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")

        x = tf.to_float(self.states) / 255.0

        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_network(x, add_summaries=(not reuse))

        with tf.variable_scope("value_net"):
            self.logits = tf.layers.dense(fc1, 1, activation=None)
            # squeeze는 1인 차원(행렬)을 날림. => [1, 2, 3] squeeze => [2, 3]
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = { "logits": self.logits }

            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss.name, self.loss)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=tf.train.get_global_step())

            var_scope_name = tf.get_variable_scope().name
            summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
            summaries = [s for s in summary_ops if var_scope_name in s.name]
            self.summaries = tf.summary.merge(summaries)
