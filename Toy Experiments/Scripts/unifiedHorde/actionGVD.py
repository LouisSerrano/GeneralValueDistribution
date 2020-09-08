import numpy as np
from Scripts.utils import ReplayBuffer
import tensorflow as tf
import pandas as pd

class ActionGVD:
    def __init__(self, agent_info={}):
        """Implements the ActionGVF for a UnifiedHorde.
           Args:
               agent_info (dictionary) : contains the GVD parameters
           Initialises:
               self.num_states (int): the dimension of the state space
               self.num_actions (int): the number of actions
               self.beta (float): the beta risk parameter
               self.horde_z (hordeModel): the gvd can have access to the whole horde model
               self.input (int): the GVD index that this gvd has access to. It is used for transition_gen.
               self.all_state_vectors (Numpy array): state vector representation of all states
               self.transition_gen (function): transition function (e.g. deterministic cumulant)
               self.transition_gen_params (dictionary): the parameters for the transition_gen function
               self.batch_size (int): the batch size
               self.buffer (ReplayBuffer): Buffer that stores transitions
               self.control (boolean): if True, the target policy is the greedy policy w.r.t the q-values
               self.average_loss (float): if True, uses average loss for QGVD
               self.policy (Numpy array): target policy
               self.file_path (String): location of the reference CSV
               """


        self.num_states = agent_info.get("num_states", 60)
        self.num_actions = agent_info.get("num_actions", 4)
        self.beta = agent_info.get("beta_risk", 0)
        self.last_state = None
        self.horde_z = agent_info.get('horde_z', None)
        self.input = agent_info.get("input", None)
        # model param
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        
        # transition gen
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})
        
        # keep track of the cumulants, gammas, states, actions and returns
        self.batch_size = agent_info.get("batch_size", 32)
        self.buffer = ReplayBuffer(self.batch_size)

        # policy
        self.control = agent_info.get("control", False)
        if self.control:
            self.policy = np.ones((self.num_states, self.num_actions))/self.num_actions
        else:
            self.average_loss = agent_info.get("average_loss", True)
            self.policy = agent_info.get("policy")

        # tensorboard
        self.mean_cumulant = tf.keras.metrics.Mean('mean_reward', dtype=tf.float32)
        self.file_path = agent_info.get('file_path', None)
        # construct reference
        if self.file_path is not None:
            self.df = pd.read_csv(self.file_path)
            self.track_list = agent_info.get("track_list", [[23, 1], [33, 0], [43, 0], [51, 1], [50, 1]])
            self.return_list = []
            for j in range(len(self.track_list)):
                self.return_list.append(self.get_ref_returns(self.track_list[j][0],
                                                             self.track_list[j][1]))
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
        action_ = np.zeros(self.num_actions, dtype = np.float32)
        action_[self.last_action] = 1.0

        # put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)

        self.last_state = state
        self.last_action = action
        
    def end(self, state, action):
        """ GVD end"""
        # get relevant feature
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]
        
        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)
        action_ = np.zeros(self.num_actions, np.float32)
        action_[self.last_action] = 1.0
        
        # put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)
        