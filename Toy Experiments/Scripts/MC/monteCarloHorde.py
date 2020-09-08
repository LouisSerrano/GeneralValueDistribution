# inherit the abstract class
import numpy as np
from Scripts.horde import BaseHorde
from .monteCarloGVD import MonteCarloGVD

class MonteCarloHorde(BaseHorde):
    """Implements the MonteCarloHorde for a RLGlueHorde environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    def horde_init(self, horde_info= {}):
        """
       Args:
           horde_info (dictionary): contains the parameters of the Separate Horde
       Initialises:
           self.policy (Numpy array): the behaviour policy of the Horde
           self.on_policy (boolean): determines if the horde is off-policy or on-policy
           self.epsilon (float): Initial epsilon value. Parameter for epsilon-greedy policy
           self.epsilon_step (float): epsilon value at current step
           self.lambd (float): parameter for softmax policy
           self.epsilon_greedy (boolean): The behaviour policy is epsilon-greedy when True
           self.soft_max (boolean): The behaviour policy is soft-max when True
           self.on_policy_GVD (int): The index of the on policy GVD
           self.policy_update_freq (int): The frequency at which we update the behaviour policy
           self.GVDs_info (dictionary): contains the the GVDs parameters
           self.GVDs (list): contains the GVDs
           self.visitation_state (Numpy array): an array that keeps track of the visited states
           self.steps (int): counts the number of steps
           self.episodes (int): counts the number of episodes
       """

        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        self.rand_generator = np.random.RandomState(horde_info.get("seed"))
        # behaviour policy
        self.policy = horde_info.get("policy")
        self.visitation_state = np.zeros(60)
        self.steps = 0

        self.GVDs_info = horde_info.get("GVDs_info")
        self.GVDs_num = len(self.GVDs_info)
        self.GVDs = []
        for i in range(self.GVDs_num):
            self.GVDs.append(MonteCarloGVD(self.GVDs_info[i]))

        self.on_policy = horde_info.get("on_policy", False)

        if self.on_policy:
            self.epsilon = horde_info.get("epsilon", 0.05)
            self.lambd = horde_info.get("lambd", 1)
            self.epsilon_greedy = horde_info.get("epsilon_greedy", False)
            self.soft_max = horde_info.get('soft_max', False)
            self.on_policy_GVF = horde_info.get("on_policy_GVF") - 1
            self.policy_update_freq = horde_info.get("policy_update_freq", 5)

    def get_state_distribution(self):
        """
        Returns:
            mu (Numpy array): the state frequencies """
        return self.visitation_state / self.steps

    def horde_start(self, state):
        """ The first method called when the experiment starts, called after
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
            rho = gvd.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
            gvd.step(state, action, rho)

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
            rho = gvd.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
            gvd.end(state, action, rho)

        self.visitation_state[state] += 1
        self.steps += 1

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
        if message == "get theta values":
            gvd = self.GVDs[index]
            return gvd.z.model(gvd.all_state_vectors)
