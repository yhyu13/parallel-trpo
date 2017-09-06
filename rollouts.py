import numpy as np
import tensorflow as tf
import multiprocessing
from utils import *
import time
import copy
from random import randint

from helper import *
from ou_noise import *

class Actor(multiprocessing.Process):
    def __init__(self, args, task_q, result_q, actor_id, monitor):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.args = args
        self.monitor = monitor
        self.noise = OUNoise(9)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.session.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs})
        # samples the guassian distribution
        act = np.clip(action_dist_mu + np.exp(action_dist_logstd)*self.noise.noise(),0.01,0.99)
        return act.ravel(), action_dist_mu, action_dist_logstd

    def run(self):

        self.env = ei(vis=False)
        
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

        # tensorflow variables (same as in model.py)
        self.observation_size = 58 #self.env.observation_space.shape[0]
        self.action_size = 9#np.prod(self.env.action_space.shape)
        self.hidden_size = 300
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)
        # tensorflow model of the policy
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.debug = tf.constant([2,2])
        with tf.variable_scope("policy-a"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
            h1 = tf.nn.elu(h1)
            h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
            h2 = tf.nn.elu(h2)
            h3 = fully_connected(h2, self.hidden_size, self.action_size, tf.random_uniform_initializer(-3e-3,3e-3), bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")
        self.action_dist_mu = tf.clip_by_value(h3,0.01,0.99)
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        var_list = tf.trainable_variables()

        self.set_policy = SetPolicyWeights(self.session, var_list)

        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)
            if next_task == 1:
                # the task is an actor request to collect experience
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == 2:
                print "kill message"
                if self.monitor:
                    self.env.monitor.close()
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                self.set_policy(next_task)
                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                time.sleep(0.1)
                self.task_q.task_done()
        return

    def rollout(self):
        obs, actions, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
        self.env.reset()
        hard_code_action = engineered_action(0.1)
        
        demo_length = 20

        for i in range(demo_length):
            self.env.step(hard_code_action)
            self.noise.noise()
            
        ob = self.env.step(hard_code_action)[0]
        s1 = self.env.step(hard_code_action)[0]
        ob = filter(process_state(ob,s1))
        
        action = hard_code_action
        
        for i in xrange(self.args.max_pathlength - 1 - demo_length):
            obs.append(ob)
            action, action_dist_mu, action_dist_logstd = self.act(ob)
            actions.append(action)
            #print(action_dist_mu)
            #print(action_dist_logstd)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)
            res = self.env.step(np.tile(action,2))
            s2 = res[0]
            s1 = filter(process_state(s1,s2))
            ob = s1
            s1 = s2
            
            if not res[2]:
                ep_reward = 0.5
            else:
                ep_reward = -10.
            if s1[2] > 0.75:
                h_reward = 0.5
            else:
                h_reward = -10.
            engineered_reward = res[1]/0.01+ep_reward+h_reward#+2*abs(s1[32]-s1[34])

            #print(engineered_reward)
            #rewards.append((res[1]))
            rewards.append((engineered_reward))
            if res[2] or i == self.args.max_pathlength - 2:
                #print(sum(rewards))
                self.noise.reset(None)
                path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                             "action_dists_mu": np.concatenate(action_dists_mu),
                             "action_dists_logstd": np.concatenate(action_dists_logstd),
                             "rewards": np.array(rewards),
                             "actions":  np.array(actions)}
                return path
                break

class ParallelRollout():
    def __init__(self, args):
        self.args = args

        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()

        self.actors = []
        self.actors.append(Actor(self.args, self.tasks, self.results, 9999, args.monitor))

        for i in xrange(self.args.num_threads-1):
            self.actors.append(Actor(self.args, self.tasks, self.results, 37*(i+3), False))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first ieration

        self.average_timesteps_in_episode = 1000

    def rollout(self):

        # keep 20,000 timesteps per update
        num_rollouts = self.args.timesteps_per_batch / self.average_timesteps_in_episode

        for i in xrange(num_rollouts):
            self.tasks.put(1)

        self.tasks.join()

        paths = []
        while num_rollouts:
            num_rollouts -= 1
            paths.append(self.results.get())

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)
        return paths

    def set_policy_weights(self, parameters):
        for i in xrange(self.args.num_threads):
            self.tasks.put(parameters)
        self.tasks.join()

    def end(self):
        for i in xrange(self.args.num_threads):
            self.tasks.put(2)
