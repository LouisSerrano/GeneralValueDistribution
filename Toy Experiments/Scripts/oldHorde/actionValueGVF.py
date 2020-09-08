from Scripts.utils import MazeTileCoder
import numpy as np


class ActionValueGVF:
    def __init__(self, agent_info={}):
        """Implements the ActionValueGVF for a Horde.
            Args:
                agent_info (dictionary) : contains the GVF parameters
            Initialises:
                self.lambd (float): lambda parameter of GQ(lambda)
                self.alpha (float): learning rate
                self.alpha_h (float): secondary learning rate
                self.num_states (int): the dimension of the state space
                self.num_actions (int): the number of actions
                self.steps (int): counts the number of steps
                self.update_freq (int): the update frequency
                self.weights (Numpy array): weight vector
                self.hWeights (Numpy array): hWeight vector
                self.eligTrace (Numpy array): eligibility trace vector
                self.all_state_vectors (Numpy array): state vector representation of all states
                self.all_state_action_vectors (Numpy array): vector representation of all state-action pairs
                self.transition_gen (function): transition function (e.g. deterministic cumulant)
                self.transition_gen_params (dictionary): the parameters for the transition_gen function
                self.buffer (ReplayBuffer): Buffer that stores transitions
                self.control (boolean): if True, the target policy is the greedy policy w.r.t the q-values
                self.policy (Numpy array): target policy
                self.q_values (Numpy array): state-action values
            """
        
        # parameters
        self.lambd = agent_info.get("lambda", 0.1)
        self.alpha = agent_info.get("alpha", 0.1)
        self.alpha_h = 0.1*self.alpha
        self.num_states = agent_info.get("num_states", 60)
        self.num_actions = agent_info.get("num_actions", 4)
        self.steps = 0
        self.update_freq = agent_info.get("update_freq", 5)
        self.current_gamma = None
        self.last_state = None
        
        # tile coding
        #self.num_tilings = agent_info.get("num_tilings", 4)
        #self.num_tiles = agent_info.get("num_tiles", 4)

        self.weights = np.zeros(self.num_states*self.num_actions)
        self.hWeights = np.zeros(self.num_states*self.num_actions)
        self.eligTrace = np.zeros(self.num_states*self.num_actions)
        
        # all_state_action_vectors
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        self.all_state_action_vectors = np.zeros((self.num_states, self.num_actions, self.num_states*self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.all_state_action_vectors[s, a, :] = self.get_state_action_vector(s,a)
        
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})
        self.control = agent_info.get("control", False)

        if not self.control:
            self.policy = agent_info.get("policy")
        else:
            self.policy = np.ones((self.num_states, self.num_actions))/self.num_actions
        self.q_values = np.zeros((self.num_states, self.num_actions))

    def get_state_action_vector(self, state, a):
        state_vector = self.all_state_vectors[state]
        state_action_vector = np.zeros(self.num_states*self.num_actions)
        state_action_vector[a*self.num_states:(a+1)*self.num_states] = state_vector
        return state_action_vector
    
    def get_values(self):
        self.set_q_values()    
        return np.sum(self.q_values*self.policy, -1)
    
    def set_q_values(self):
        self.q_values = np.sum(self.all_state_action_vectors * self.weights, -1)
        
    def get_q_values(self):
        self.set_q_values()
        return self.q_values
    
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

    def argmax(self, q_values):
        """ Args:
                q_values (Numpy array): shape (num_actions)
            Returns:
                actions (list) : a list of actions that maximises q_values
                """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        return ties
    
    def get_policy(self):
        return self.policy

    def update_greedy_policy(self):
        for s in range(self.num_states):
            self.state_update_greedy_policy(s)
            
    def update_epsilon_greedy_policy(self, epsilon):
        """ Computes the epsilon-greedy policy based on the q-values
            Args:
                lambd (float)
            Returns:
                policy (list) : the updated behaviour policy
            """
        policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            policy[s,:] = self.state_update_epsilon_greedy_policy(s, epsilon)    
        return policy
        
    def state_update_greedy_policy(self, state):
        arg = self.argmax(self.q_values[state])
        for a in range(self.num_actions):
            self.policy[state, a] = 0
            if a in arg:
                self.policy[state, a] = 1/len(arg)
                
    def state_update_epsilon_greedy_policy(self, state, epsilon):
        arg_max = self.argmax(self.q_values[state])
        pi = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            if (action in arg_max):
                pi[action] = (1-epsilon)/len(arg_max) + epsilon/4
            else :
                pi[action] = epsilon/4
        return pi

    
    def start(self, state, action):
        """GVD start"""
        self.last_state = state
        self.last_action = action
        self.eligTrace = np.zeros(self.num_states*self.num_actions)
        return action
    

    def step(self, state, action, rho = 1.0):
        """GVD step"""
        # get relevant feature
        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)

        # retrieve the values
        last_vector = self.get_state_action_vector(self.last_state, self.last_action)
        self.last_value = np.sum(self.weights*last_vector)
        current_vector = np.zeros(self.num_states*self.num_actions)
        
        # mean current vector
        for a in range(4):
            current_vector += self.policy[state][a]*self.get_state_action_vector(state, a) 
        current_value = np.sum(self.weights*current_vector)
        
        # GQ lambda update
        self.delta = cumulant + (gamma * current_value - self.last_value)
        self.eligTrace = rho * self.lambd * self.last_gamma * self.eligTrace + last_vector
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.sum(self.eligTrace * self.hWeights)*current_vector)
        self.hWeights += self.alpha_h * (self.delta*self.eligTrace - np.sum(self.hWeights*last_vector)*last_vector)
        
       # update q_values
        if self.control:
            if self.steps % self.update_freq == 0:
                self.set_q_values()
                self.update_greedy_policy()
            
        self.last_gamma = gamma
        self.last_state = state
        self.last_action = action
        self.steps += 1
        
    def end(self, state, action, rho = 1.0):
        """GVD end"""
        # get relevant feature
        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)

        # retrieve the values
        last_vector = self.get_state_action_vector(self.last_state, self.last_action)
        self.last_value = np.sum(self.weights*last_vector)
        current_vector = np.zeros(self.num_states*self.num_actions)
        
        # mean current vector
        for a in range(self.num_actions):
            current_vector += self.policy[state][a]*self.get_state_action_vector(state, a) 
        current_value = np.sum(self.weights*current_vector)
        
        
        # GQ lambda update
        self.delta = cumulant + (gamma * current_value - self.last_value)
        self.eligTrace = rho * self.lambd * self.last_gamma * self.eligTrace + last_vector
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.sum(self.eligTrace * self.hWeights)*current_vector)
        self.hWeights += self.alpha_h * (self.delta*self.eligTrace - np.sum(self.hWeights*last_vector)*last_vector)

        # update q_values
        if self.control:
            self.set_q_values()
            self.update_greedy_policy()
        
        self.last_gamma = gamma
        self.last_state = state
        self.steps += 1