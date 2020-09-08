# inherit the abstract class
import numpy as np
from Scripts.horde import BaseHorde
from .separateActionGVD import SeparateActionGVD
import tensorflow as tf
import scipy
import datetime

class SeparateHorde(BaseHorde):
    """Implements the SeparateHorde for an RLGlueHorde environment.
    Note:
        horde_init, horde_start, horde_step, horde_end, horde_cleanup, and
        horde_message are required methods.
    """
    def horde_init(self, horde_info= {}):
        """
        Args:
            horde_info (dictionary): contains the parameters of the Separate Horde
        Initialises:
            self.policy (Numpy array): the behaviour policy of the Horde
            self.visitation_state (Numpy array): an array that keeps track of the visited states
            self.steps (int): counts the number of steps
            self.episodes (int): counts the number of episodes
            self.GVDs_info (list): a list containing the GVDs information
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
        self.rand_generator = np.random.RandomState(horde_info.get("seed"))
        # behaviour policy
        self.policy = horde_info.get("policy")

        self.visitation_state = np.zeros(self.policy.shape[0])
        self.steps = 0
        self.episodes = 0

        self.GVDs_info = horde_info.get("GVDs_info")
        self.GVDs_num = len(self.GVDs_info)
        self.GVDs = []
        for i in range(self.GVDs_num):
            self.GVDs_info[i]["index"] = i
            self.GVDs.append(SeparateActionGVD(self.GVDs_info[i]))

        self.on_policy = horde_info.get("on_policy", False)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.on_policy:
            sub_dir = 'on_policy/'
        else:
            sub_dir = 'off_policy/'
        self.log_dir = 'logs/gradient_tape/' + sub_dir + current_time + '/train'
        self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.on_policy:
            self.epsilon = horde_info.get("epsilon", 0.1)
            self.epsilon_step = self.epsilon
            self.lambd = horde_info.get("lambd", 1)
            self.epsilon_greedy = horde_info.get("epsilon_greedy", False)
            self.soft_max = horde_info.get('soft_max', False)
            self.on_policy_GVD = horde_info.get("on_policy_GVD") - 1
            self.policy_update_freq = horde_info.get("policy_update_freq", 1000)

    def epsilon_scheduler(self):
        """
        Sets the value of self.epsilon_step
        """

        m = self.episodes//25
        self.epsilon_step = np.max([self.epsilon/(1 + np.sqrt(m)), 0.1])

    def get_state_distribution(self):
        """
        Returns:
            mu (Numpy array): the state frequencies """
        return self.visitation_state / self.steps

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
            # self.GVFs.update({name : gvf})

        self.last_state = state
        self.last_action = action
        self.visitation_state[state] += 1
        self.steps += 1

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
            self.tensorboard_step(gvd, j)

        # policy update
        if self.on_policy:
            if self.steps % self.policy_update_freq == 0:
                gvd = self.GVDs[self.on_policy_GVD]
                if self.epsilon_greedy:
                    self.policy = gvd.update_epsilon_greedy_policy(self.epsilon_step)
                elif self.soft_max:
                    self.policy = gvd.update_soft_max_policy(self.lambd)

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

        # GVDs end
        for j in range(self.GVDs_num):
            gvd = self.GVDs[j]
            gvd.end(state, action)
            self.tensorboard_end(gvd, j)

        # policy update
        if self.on_policy:
            gvd = self.GVDs[self.on_policy_GVD]
            if self.epsilon_greedy:
                self.epsilon_scheduler()
                self.policy = gvd.update_epsilon_greedy_policy(self.epsilon_step)
            elif self.soft_max:
                self.policy = gvd.update_soft_max_policy(self.lambd)

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
                tf.summary.scalar('cumulant_gvd_{}'.format(j), gvd.mean_cumulant.result()*100, step=self.steps // 100)
                gvd.mean_cumulant.reset_states()

        if gvd.file_path is not None:
            if self.steps % 1000 == 0:
                for i in range(len(gvd.track_list)):
                    last_s, last_a = gvd.track_list[i][0], gvd.track_list[i][1]
                    state_vector = gvd.all_state_vectors[last_s]
                    w = scipy.stats.wasserstein_distance(gvd.return_list[i], gvd.z.predict(state_vector)[0][last_a])
                    with self.writer.as_default():
                        tf.summary.scalar("Wasserstein_gvd_{}_s_{}_a_{}".format(j, last_s, last_a), w, step=self.steps // 1000)

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
            tf.summary.scalar('loss' + str(j), gvd.z.train_loss.result(), step=self.episodes)
            gvd.z.train_loss.reset_states()

        if self.steps % 100 == 0:
            with self.writer.as_default():
                tf.summary.scalar('cumulant_gvd_{}'.format(j), gvd.mean_cumulant.result()*100, step=self.steps // 100)
                gvd.mean_cumulant.reset_states()

        if gvd.file_path is not None:
            if self.steps % 1000 == 0:
                for i in range(len(gvd.track_list)):
                    last_s, last_a = gvd.track_list[i][0], gvd.track_list[i][1]
                    state_vector = gvd.all_state_vectors[last_s]
                    w = scipy.stats.wasserstein_distance(gvd.return_list[i], gvd.z.predict(state_vector)[0][last_a])
                    with self.writer.as_default():
                        tf.summary.scalar("Wasserstein_gvd_{}_s_{}_a_{}".format(j, last_s, last_a), w,
                                          step=self.steps // 1000)


    def horde_cleanup(self):
        """Cleanup done after the agent ends."""

    def horde_message(self, message, index):
        index = max(index - 1, 0)
        if message == "get state values":
            gvd = self.GVDs[index]
            pi = gvd.policy
            theta = gvd.z.model(gvd.all_state_vectors)
            q = np.mean(theta, -1)
            return np.sum(q * pi, -1)
        if message == "get state distribution":
            return self.visitation_state / self.steps
        if message == "get action values":
            gvd = self.GVDs[index]
            theta = gvd.z.model(gvd.all_state_vectors)
            q = np.mean(theta, -1)
            return q
        if message == "get beta values":
            gvd = self.GVDs[index]
            theta = gvd.z.model(gvd.all_state_vectors)
            q_beta = 1/gvd.beta * np.log(np.mean(np.exp(theta*gvd.beta), -1))
            return q_beta

        if message == "get theta values":
            gvd = self.GVDs[index]
            return gvd.z.model(gvd.all_state_vectors)
