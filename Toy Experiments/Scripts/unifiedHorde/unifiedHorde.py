import numpy as np
import tensorflow as tf
import random
import scipy
from Scripts.horde import BaseHorde
from Scripts.unifiedHorde.actionGVD import ActionGVD
from Scripts.unifiedHorde.hordeModel import HordeModel
import datetime

class UnifiedHorde(BaseHorde):
    """Implements the UnifiedHorde for an RLGlueHorde environment.
        Note:
            horde_init, horde_start, horde_step, horde_end, horde_cleanup, and
            horde_message are required methods.
        """

    def horde_init(self, horde_info= {}):
        """
        Args:
            horde_info (dictionary): contains the parameters of the Unified Horde
        Initialises:
            self.on_policy (boolean): determines if the horde is off-policy or on-policy
            self.policy (Numpy array): the behaviour policy of the Horde
            self.num_states (int): the dimension of the state space
            self.num_actions (int) : the number of actions
            self.all_state_vectors (Numpy array): the vector representation for all states
            self.visitation_state (Numpy array): keeps track of the visited states
            self.steps (int): counts the number of steps
            self.episodes (int): counts the number of episodes
            self.num_quantiles (int): the number of quantiles
            self.tau (tf array): the quantile levels
            self.kappa (float): quantile huber loss parameter
            self.eta (float): learning rate
            self.learning_rate_scheduler (boolean): if True, the learning rate is scheduled to decay over steps
            self.GVDs_info (list): a list containing the GVDs information
            self.z (HordeModel): prediction Network
            self.z_target (HordeModel): target Network
            self.batch_size (int): batch size
            self.update_freq (int): the update frequency for the target network
            self.replay_freq (int): the replay frequency for the prediction network
            self.GVDs (list): a list containing the GVDs objects
            self.log_dir (String): directory for tensorboard directory
            self.writer (tf File writer)
            self.on_policy (boolean): determines if the horde is off-policy or on-policy
            self.epsilon (float): Initial epsilon value. Parameter for epsilon-greedy policy
            self.epsilon_step (float): epsilon value at current step
            self.lambd (float): parameter for softmax policy
            self.epsilon_greedy (boolean): The behaviour policy is epsilon-greedy when True
            self.soft_max (boolean): The behaviour policy is soft-max when True
            self.on_policy_GVD (int): The index of the on policy GVD
            self.policy_update_freq (int): The frequency to which we update the behaviour policy
        """

        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        random.seed(horde_info.get("seed"))
        self.rand_generator = np.random.RandomState(horde_info.get("seed"))

        # behaviour policy
        self.on_policy = horde_info.get("on_policy", False)
        self.policy = horde_info.get("policy")

        # environment
        self.num_states = horde_info.get("num_states", self.policy.shape[0])
        self.num_actions = horde_info.get("num_actions", 4)
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        self.visitation_state = np.zeros(self.num_states)
        self.steps = 0
        self.episodes = 0
        
        # GVD parameters        
        self.num_quantiles = horde_info.get("num_quantiles", 51)
        self.tau = tf.constant((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), dtype = tf.float32)
        self.kappa = horde_info.get("kappa", 1)
        self.eta = horde_info.get("eta", 0.001)
        self.GVDs_info = horde_info.get("GVDs_info")
        self.GVDs_num = len(self.GVDs_info)
        self.learning_rate_scheduler = horde_info.get("lr_scheduler", False)

        if self.learning_rate_scheduler:
            initial_learning_rate = self.eta
            self.eta = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=20000,
                decay_rate=0.5,
                staircase=True)

        
        # create the model
        self.z = HordeModel(self.num_states, self.GVDs_num, self.num_actions, self.num_quantiles, self.kappa, self.eta)
        self.z_target = HordeModel(self.num_states, self.GVDs_num, self.num_actions, self.num_quantiles, self.kappa, self.eta)
        self.batch_size = horde_info.get("batch_size", 32)
        self.update_freq = horde_info.get("update_freq", 50)
        self.replay_freq = horde_info.get("replay_freq", 5)

        # create GVDs 
        self.GVDs = []
        for i in range(self.GVDs_num):
        # give access to the model for GVDs that require the theta values of other GVDs 
            if self.GVDs_info[i].get('input') != None:
                self.GVDs_info[i]["horde_z"]= self.z
            self.GVDs_info[i]["batch_size"] = self.batch_size
            self.GVDs.append(ActionGVD(self.GVDs_info[i]))

        # tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.on_policy:
            sub_dir = 'on_policy/'
        else:
            sub_dir = 'off_policy/'
        self.log_dir = 'logs/gradient_tape/' + sub_dir + current_time + '/train'
        self.writer = tf.summary.create_file_writer(self.log_dir)

        # get the params of the on_policy GVD
        if self.on_policy:
            self.on_policy_GVD = horde_info.get("on_policy_GVD") - 1
            self.epsilon = horde_info.get("epsilon", 0.1)
            self.lambd = horde_info.get("lambd", 1)
            self.epsilon_greedy = horde_info.get("epsilon_greedy", False)
            self.soft_max = horde_info.get('soft_max', False)
            self.policy_update_freq = horde_info.get("policy_update_freq", 100)
        
    def target_update(self):
        """ Updates the weights of the target network"""
        weights = self.z.model.get_weights()
        self.z_target.model.set_weights(weights)
    
    def get_state_distribution(self):
        """
        Returns:
            mu (Numpy array): the state frequencies """
        return self.visitation_state / self.steps
    
    def replay(self, index):
        """Performs a replay step for the GVD located at the index position in self.GVDs
        Args:
            index (int): the location of a GVD in the list self.GVDs
        """
        gvd = self.GVDs[index]
        last_state_vectors, last_actions, cumulants, gammas, current_states, current_state_vectors, next_actions = gvd.buffer.sample()
        z = self.z_target.gvd_model(current_state_vectors, index)
        
        # control training
        if gvd.control:
            theta = np.zeros((self.batch_size, self.num_quantiles))
            # risk neutral
            if gvd.beta == 0:
                next_actions = np.argmax(np.mean(z, axis=2), axis=1)
            # risk sensitive
            else:
                next_actions = np.argmax(1 / gvd.beta * np.log(np.mean(np.exp(z * gvd.beta), axis=2)), axis=1)

            for i in range(self.batch_size):
                theta[i,:] = cumulants[i] + gammas[i] * z[i][next_actions[i]]
            
            self.z.train(last_state_vectors, theta, last_actions, index)

        # evaluation training
        else:
            # average loss policy requires different training
            if gvd.average_loss:
                thetas = np.zeros((self.batch_size, self.num_actions, self.num_quantiles))
                pi = tf.constant(gvd.policy[current_states], tf.float32)
                for i in range(self.batch_size):
                    thetas[i,:] = cumulants[i] + gammas[i] * z[i]
                self.z.avg_loss_train(last_state_vectors, thetas, last_actions, index, pi)
            # one-sample training
            else:
                theta = np.zeros((self.batch_size, self.num_quantiles))
                for i in range(self.batch_size):
                    next_action = np.random.choice(range(self.policy.shape[1]), p = gvd.policy[current_states[i]])
                    theta[i, :] = cumulants[i] + gammas[i] * z[i][next_action]
                self.z.train(last_state_vectors, theta, last_actions, index)


    def horde_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Runs sequentially the start method for every GVD.

        Args:
            state (int): the state observation from the environment's env_start function.
        Returns:
            The first action the horde takes.
        """
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

        # GVDs start
        for j in range(self.GVDs_num):
            gvd = self.GVDs[j]
            gvd.start(state, action)

        self.last_state = state
        self.last_action = action
        self.visitation_state[state] += 1
        self.steps +=1
        
        return action

    def horde_step(self, state):
        """A step taken by the horde. The horde sequentially runs the step method for every GVD.
        Args:
            state (int): the state observation from the
                environment's step based, where the horde ended up after the
                last step
        Returns:
            The action the horde is taking.
        """
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
                
        # GVDs step
        for j in range(self.GVDs_num):
            gvd = self.GVDs[j]
            gvd.step(state, action)
            if self.steps > self.batch_size:
                if self.steps % self.replay_freq == 0:
                    self.replay(j)
            if self.steps % self.update_freq == 0:
                self.target_update()
            self.tensorboard_step(gvd, j)

        if self.on_policy:
            if self.steps % self.policy_update_freq == 0:
                if self.epsilon_greedy:
                    self.policy = self.update_epsilon_greedy_policy(self.epsilon)
                elif self.soft_max:
                    self.policy = self.update_soft_max_policy(self.lambd)

        self.last_state = state
        self.last_action = action
        self.visitation_state[state] += 1
        self.steps += 1
        
        return self.last_action

    def horde_end(self, state):
        """Run when the agent terminates. The horde sequentially runs the end method for every GVD.
        """
        # the action does not matter actually
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        
         # GVDs step
        for j in range(self.GVDs_num):
            gvd = self.GVDs[j]
            gvd.end(state, action)
            if self.steps > self.batch_size:
                if self.steps % self.replay_freq == 0:
                    self.replay(j)
                if self.steps % self.update_freq == 0:
                    self.target_update()
            self.tensorboard_end(gvd, j)

        if self.on_policy:
            if self.steps % self.policy_update_freq == 0:
                if self.epsilon_greedy:
                    self.policy = self.update_epsilon_greedy_policy(self.epsilon)
                elif self.soft_max:
                    self.policy = self.update_soft_max_policy(self.lambd)


        self.visitation_state[state] += 1
        self.steps += 1
        self.episodes += 1

    def tensorboard_step(self, gvd, j):
        """
           Called with horde step.
           Args:
               gvd (GVD-like object)
               j : the index of the corresponding GVD in self.GVDs
           Writes:
               the sum of cumulants over the last 100 steps
               the wasserstein distance between theta[s,a] and the monte carlo samples
               """
        if self.steps % 100 == 0:
            with self.writer.as_default():
                tf.summary.scalar('cumulant_gvd_{}'.format(j), gvd.mean_cumulant.result() * 100, step=self.steps // 100)
                gvd.mean_cumulant.reset_states()

        if gvd.file_path is not None:
            if self.steps % 1000 == 0:
                for i in range(len(gvd.track_list)):
                    last_s, last_a = gvd.track_list[i][0], gvd.track_list[i][1]
                    state_vector = gvd.all_state_vectors[last_s]
                    state_vector = np.reshape(state_vector, (1,-1))
                    w = scipy.stats.wasserstein_distance(gvd.return_list[i], self.z.gvd_model(state_vector, j)[0][last_a])
                    with self.writer.as_default():
                        tf.summary.scalar("Wasserstein_gvd_{}_s_{}_a_{}".format(j, last_s, last_a), w,
                                          step=self.steps // 1000)

    def tensorboard_end(self, gvd, j):
        """
        Called with horde end.
        Args:
            gvd (GVD-like object)
            j : the index of the corresponding GVD in self.GVDs
        Writes:
            the average loss over the episode
            the sum of cumulants over the last 100 steps
            the wasserstein distance between theta[s,a] and the monte carlo samples
                """
        with self.writer.as_default():
            tf.summary.scalar('loss', self.z.train_loss.result(), step=self.episodes)
            self.z.train_loss.reset_states()

        if self.steps % 100 == 0:
            with self.writer.as_default():
                tf.summary.scalar('cumulant_gvd_{}'.format(j), gvd.mean_cumulant.result() * 100, step=self.steps // 100)
                gvd.mean_cumulant.reset_states()

        if gvd.file_path is not None:
            if self.steps % 1000 == 0:
                for i in range(len(gvd.track_list)):
                    last_s, last_a = gvd.track_list[i][0], gvd.track_list[i][1]
                    state_vector = gvd.all_state_vectors[last_s]
                    state_vector = np.reshape(state_vector, (1, -1))
                    w = scipy.stats.wasserstein_distance(gvd.return_list[i], self.z.gvd_model(state_vector, j)[0][last_a])
                    with self.writer.as_default():
                        tf.summary.scalar("Wasserstein_gvd_{}_s_{}_a_{}".format(j, last_s, last_a), w,
                                          step=self.steps // 1000)

    #def set_q_values(self, index):
    #    quantiles = self.z.gvd_model(self.all_state_vectors, index)
    #    q_values = np.mean(quantiles, -1)
    #    self.q_values = q_values

    def get_q_values(self, index):
        """
        Args:
            index (int): the index of the GVD in self.GVDs
        Returns:
            q_values (Numpy array): the state-action values of the corresponding GVD
        """
        quantiles = self.z.gvd_model(self.all_state_vectors, index)
        q_values = np.mean(quantiles, -1)
        return q_values

    #def set_q_beta(self, index):
    #    quantiles = self.z.gvd_model(self.all_state_vectors, index)
    #    q_beta = 1 / self.beta * np.log(np.mean(np.exp(quantiles * self.beta), -1))
    #    self.q_beta = q_beta

    def get_q_beta(self, index, beta):
        """
        Args:
            index (int): the index of the GVD in self.GVDs
            beta (float): the GVD risk parameter
        Returns:
            q_values (Numpy array): the q_beta values of the corresponding GVD
        """
        quantiles = self.z.gvd_model(self.all_state_vectors, index)
        q_beta = 1 / beta * np.log(np.mean(np.exp(quantiles * beta), -1))
        return q_beta

    def update_soft_max_policy(self, lambd):
        """ Computes the soft-max policy based on the target-network of the on-policy GVD
            Args:
                lambd (float)
            Returns:
                policy (list) : the updated behaviour policy
            """
        gvd = self.GVDs[self.on_policy_GVD]
        if gvd.beta == 0:
            q_values = self.get_q_values(self.on_policy_GVD)
        else:
            q_values = self.get_q_beta(self.on_policy_GVD, gvd.beta)
        m = np.max(lambd * q_values, -1, keepdims=True)
        e = np.exp(lambd * q_values - m)
        e_sum = np.expand_dims(np.sum(e, -1), -1)
        policy = e / e_sum

        return policy

    def update_epsilon_greedy_policy(self, epsilon):
        """ Computes the epsilon-greedu policy based on the target-network of the on-policy GVD
            Args:
                epsilon (float)
            Returns:
                policy (list) : the updated behaviour policy
            """
        policy = np.zeros((self.num_states, self.num_actions))
        gvd = self.GVDs[self.on_policy_GVD]
        if gvd.beta == 0:
            q_values = self.get_q_values(self.on_policy_GVD)
        else:
            q_values = self.get_q_beta(self.on_policy_GVD, gvd.beta)
        for s in range(self.num_states):
            arg = self.argmax(q_values[s])
            for a in range(self.num_actions):
                if a in arg:
                    policy[s, a] = (1 - epsilon) / len(arg) + epsilon / self.num_actions
                else:
                    policy[s, a] = epsilon / self.num_actions
        return policy

    def argmax(self, q_values):
        """ Args:
              q_values (Numpy array): shape (num_actions)
            Returns:
                actions (list) : a list of actions that maximises q_values
              """
        arg = np.argwhere(q_values == np.amax(q_values))
        return arg.flatten().tolist()
            
    def horde_cleanup(self):
        """Cleanup done after the agent ends."""

    def horde_message(self, message, index):
        index = max(index-1, 0)
        if message == "get state values":
            pi = self.GVDs[index].policy
            theta = self.z.gvd_model(self.all_state_vectors, index)
            q = np.mean(theta, -1)
            return np.sum(q*pi, -1)
        if message == "get state distribution":
            return self.visitation_state/self.steps
        if message == "get action values":
            theta = self.z.gvd_model(self.all_state_vectors, index)
            q = np.mean(theta, -1)
            return q
        if message == "get theta values":
            return self.z.gvd_model(self.all_state_vectors, index)