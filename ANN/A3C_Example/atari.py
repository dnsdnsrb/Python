import numpy as np
import tensorflow as tf

class AtariEnvWrapper(object):

    #목숨 수 관련인 듯?

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *args, **kwargs):
        lives_before = self.env.ale.lives()
        next_state, reward, done, info = self.env.step(*args, **kwargs)
        lives_after = self.env.ale.lives()

        if lives_before > lives_after:
            done = True

        reward = max(min(reward, 1), -1)

        return next_state, reward, done, info

def atari_make_inital_state(state):
    return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
    return np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

# 전처리 관련 클래스
class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, { self.input_state: state})
