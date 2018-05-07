import os
import time

import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from A3C_Example import atari
from A3C_Example.estimators import PolicyEstimator
from A3C_Example.worker import make_copy_params_op


class PolicyMonitor(object):

    def __init__(self, env, policy_net, summary_writer, saver=None):

        self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        self.video_dir = os.path.abspath(self.video_dir)

        # Monitor는 widnows에선 사용 불가능한 것으로 보임.
        self.env = Monitor(env, directory=self.video_dir, video_callable=lambda x: True, resume=True)
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        self.sp = atari.StateProcessor()

        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

        try:
            print("video_dir", self.video_dir)
            os.makedirs(self.video_dir)
        except FileExistsError:
            print("file exists.")
            pass

        with tf.variable_scope("policy_eval"):
            self.policy_net = PolicyEstimator(policy_net.num_outputs)

        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    def _policy_net_predict(self, state, sess):
        feed_dict = { self.policy_net.states: [state] }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["probs"][0]

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            global_step, _ = sess.run([tf.train.get_global_step(), self.copy_params_op])

            done = False
            state = atari.atari_make_inital_state(self.sp.process(self.env.reset()))
            total_reward = 0.0
            episode_length = 0
            while not done:
                action_probs = self._policy_net_predict(state, sess)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                next_state = atari.atari_make_next_state(state, self.sp.process(next_state))
                total_reward += reward
                episode_length += 1
                state = next_state

            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            if self.saver is not None:
                self.saver.save(sess, self.checkpoint_path)

            tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}".format(global_step, total_reward, episode_length))

            return total_reward, episode_length

    def continous_eval(self, eval_every, sess, coord):
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            print("monitor error")
            return