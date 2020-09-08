import numpy as np
import tensorflow as tf
from Scripts.utils import ReplayBuffer
from .separateModel import SeparateModel
import pandas as pd
import os
import scipy

class SeparateActionGVD:
    def __init__(self, agent_info={}):
        """Implements the SeparateActionGVD for a SeparateHorde.
        Args:
            agent_info (dictionary) : contains the GVD parameters
        Initialises:
            self.num_states (int): the dimension of the state space
            self.num_actions (int): the number of actions
            self.batch_size (int): the batch size
            self.update_freq (int): the update frequency for the target network
            self.replay_freq (int): the replay frequency for the prediction network
            self.epsilon_adam (float): the epsilon parameter for the Adam Optimize
            self.kappa (float): the parameter of the quantile huber loss
            self.num_quantiles (int): the number of quantiles
            self.eta (float): the learning rate
            self.learning_rate_scheduler (boolean): if True, the learning rate is scheduled to decay over steps
            self.z (SeparateModel): prediction network
            self.z_target (SeparateModel): target network
            self.file_path (String): location of the reference CSV
            self.all_state_vectors (Numpy array): state vector representation of all states
            self.transition_gen (function): transition function (e.g. deterministic cumulant)
            self.transition_gen_params (dictionary): the parameters for the transition_gen function
            self.buffer (ReplayBuffer): Buffer that stores transitions
            self.control (boolean): if True, the target policy is the greedy policy w.r.t the q-values
            self.beta (float): the beta risk parameter
            self.average_loss (float): if True, uses average loss for QGVD
            self.policy (Numpy array): target policy
        """

        self.num_states = agent_info.get("num_states", 60)
        self.num_actions = agent_info.get("num_actions", 4)
        # model param
        self.batch_size = agent_info.get("batch_size", 32)
        self.update_freq = agent_info.get("update_freq", 100)
        self.replay_freq = agent_info.get("replay_freq", 50)
        self.epsilon_adam = agent_info.get("epsilon_adam", 1e-07)
        self.kappa = agent_info.get("kappa", 1)
        self.num_quantiles = agent_info.get("num_quantiles", 51)
        self.eta = agent_info.get("eta", 0.001)
        self.learning_rate_scheduler = agent_info.get("lr_scheduler", False)

        if self.learning_rate_scheduler:
            initial_learning_rate = self.eta
            self.eta = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True)

        self.z = SeparateModel(self.num_states, self.num_actions, self.num_quantiles, self.kappa, self.eta, self.epsilon_adam)
        self.z_target = SeparateModel(self.num_states, self.num_actions, self.num_quantiles, self.kappa, self.eta, self.epsilon_adam)
        # tensorboard
        self.mean_cumulant = tf.keras.metrics.Mean('mean_reward', dtype=tf.float32)
        self.file_path = agent_info.get('file_path', None)
        # construct reference
        if self.file_path is not None:
            self.df = pd.read_csv(self.file_path)
            self.track_list = agent_info.get("track_list", [[23,1], [33,0], [43,0], [51,1], [50,1]])
            self.return_list = []
            for j in range(len(self.track_list)):
                self.return_list.append(self.get_ref_returns(self.track_list[j][0],
                                                             self.track_list[j][1]))
        self.last_state = None
        self.all_state_vectors = np.diag(np.ones(self.num_states))

        # transition gen
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})

        # keep track of the cumulants, gammas, states, actions and returns
        self.buffer = ReplayBuffer(self.batch_size)

        # policy
        self.control = agent_info.get("control", False)
        self.beta = agent_info.get("beta", 0)
        if self.control:
            self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        else:
            self.average_loss = agent_info.get("average_loss", True)
            self.policy = agent_info.get("policy")

        self.steps = 0

    # retrieve the returns from Monte Carlo
    def get_ref_returns(self, last_state, last_action):
        """
        Args:
            last_state (int): the last state
            last_action (int): the last action
        Returns:
            the list of returns for (last_state, last_obs) in the reference DataFrame
        """
        mask = (self.df["last_state"] == last_state) & (self.df["last_action"] == last_action)
        return np.array(self.df.loc[mask, "return"])

    def transition_gen(self, last_state, last_action, state):
        """
       Args:
           last_state (int): the last state
           last_action (int): the last action
           state (int): the current state
       Returns:
           (cumulant, gamma)
       """
        return self.transition_gen_(self, last_state, last_action, state, self.transition_gen_params)

    def target_update(self):
        """ Updates the weights of the target network"""
        weights = self.z.model.get_weights()
        self.z_target.model.set_weights(weights)

    def argmax(self, q_values):
        """ Args:
                q_values (Numpy array): shape (num_actions)
            Returns:
                actions (list) : a list of actions that maximises q_values
                """
        arg = np.argwhere(q_values == np.amax(q_values))
        return arg.flatten().tolist()

    def set_q_values(self):
        quantiles = self.z.predict(self.all_state_vectors)
        q_values = np.mean(quantiles, -1)
        self.q_values = q_values

    def get_q_values(self):
        self.set_q_values()
        return self.q_values

    def set_q_beta(self):
        quantiles = self.z.predict(self.all_state_vectors)
        q_beta = 1 / self.beta * np.log(np.mean(np.exp(quantiles * self.beta), -1))
        self.q_beta = q_beta

    def get_q_beta(self):
        self.set_q_beta()
        return self.q_beta

    def update_soft_max_policy(self, lambd):
        """ Computes the soft-max policy based on the target-network
            Args:
                lambd (float)
            Returns:
                policy (list) : the updated behaviour policy
            """
        quantiles = self.z_target.predict(self.all_state_vectors)
        if self.beta == 0:
            q_values = np.mean(quantiles, -1)
        else:
            q_values = 1 / self.beta * np.log(np.mean(np.exp(quantiles * self.beta), -1))

        m = np.max(lambd * q_values, -1, keepdims=True)
        e = np.exp(lambd * q_values - m)
        e_sum = np.expand_dims(np.sum(e, -1), -1)
        policy = e / e_sum

        return policy

    def update_epsilon_greedy_policy(self, epsilon):
        """ Computes the epsilon-greedy policy based on the target-network
            Args:
                epsilon (float)
            Returns:
                policy (list) : the updated behaviour policy
            """
        policy = np.zeros((self.num_states, self.num_actions))
        quantiles = self.z_target.predict(self.all_state_vectors)
        if self.beta == 0:
            q_values = np.mean(quantiles, -1)
        else:
            q_values = 1 / self.beta * np.log(np.mean(np.exp(quantiles * self.beta), -1))
        for s in range(self.num_states):
            arg = self.argmax(q_values[s])
            for a in range(self.num_actions):
                if a in arg:
                    policy[s, a] = (1 - epsilon) / len(arg) + epsilon / self.num_actions
                else:
                    policy[s, a] = epsilon / self.num_actions
        return policy

    def start(self, state, action):
        """GVD start"""
        self.last_state = state
        self.last_action = action
        return action

    def step(self, state, action):
        """GVD step"""
        # get vector representation
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]

        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)
        self.mean_cumulant(cumulant)

        action_ = np.zeros(self.num_actions, np.float32)
        action_[self.last_action] = 1.0
        # put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)
        # replay
        if self.steps > self.batch_size:
            if self.steps % self.replay_freq == 0:
                self.replay()

        if self.steps % self.update_freq == 0:
            self.target_update()

        self.last_state = state
        self.last_action = action
        self.steps += 1


    def end(self, state, action):
        """GVD end"""
        # get vector representation
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]

        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)
        self.mean_cumulant(cumulant)
        action_ = np.zeros(self.num_actions, np.float32)
        action_[self.last_action] = 1.0

        # put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)

        # replay
        if self.steps > self.batch_size:
            if self.steps % self.replay_freq == 0:
                self.replay()

        if self.steps % self.update_freq == 0:
            self.target_update()

        self.steps += 1

    def replay(self):
        """Performs a replay step"""
        # sample from the buffer
        last_state_vectors, last_actions, cumulants, gammas, current_states, current_state_vectors, next_actions = self.buffer.sample()
        z = self.z_target.model(current_state_vectors)

        # control training
        if self.control:
            theta = np.zeros((self.batch_size, self.num_quantiles))
            # risk neutral
            if self.beta == 0:
                next_actions = np.argmax(np.mean(z, axis=2), axis=1)
            # risk sensitive
            else:
                next_actions = np.argmax(1 / self.beta * np.log(np.mean(np.exp(z*self.beta), axis=2)), axis=1)
            for i in range(self.batch_size):
                theta[i, :] = cumulants[i] + gammas[i] * z[i][next_actions[i]]
            self.z.train(last_state_vectors, theta, last_actions)

        # evaluation training
        else:
            # average loss policy requires different training
            if self.average_loss:
                thetas = np.zeros((self.batch_size, self.num_actions, self.num_quantiles))
                pi = tf.constant(self.policy[current_states], tf.float32)
                for i in range(self.batch_size):
                    thetas[i, :] = cumulants[i] + gammas[i] * z[i]
                self.z.avg_loss_train(last_state_vectors, thetas, last_actions, pi)
            # one-sample training
            else:
                theta = np.zeros((self.batch_size, self.num_quantiles))
                for i in range(self.batch_size):
                    next_action = np.random.choice(range(self.policy.shape[1]), p=self.policy[current_states[i]])
                    theta[i, :] = cumulants[i] + gammas[i] * z[i][next_action]
                self.z.train(last_state_vectors, theta, last_actions)
