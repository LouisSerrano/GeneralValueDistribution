from Scripts.utils import MazeTileCoder
import numpy as np


class ValueGVF:
    def __init__(self, agent_info={}):
        self.num_states = agent_info.get("num_states", 60)
        self.lambd = agent_info.get("lambda", 0.1)
        self.alpha = agent_info.get("alpha", 0.1)
        self.alpha_h =  0.1*self.alpha
        self.gamma = agent_info.get("gamma", 0.95)
        self.last_gamma = self.gamma
        self.current_gamma = None
        self.last_state = None
        
        #self.num_tilings = agent_info.get("num_tilings", 4)
        #self.num_tiles = agent_info.get("num_tiles", 4)
        #self.tc = MazeTileCoder(num_tilings=self.num_tilings, num_tiles=self.num_tiles)
        #self.iht_size = self.tc.iht_size

        self.weights = np.zeros(self.num_states)
        self.hWeights = np.zeros(self.num_states)
        self.eligTrace = np.zeros(self.num_states)

        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})
        
        self.policy = agent_info.get("policy")
        self.values = np.zeros((6,10))
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        
    def transition_gen(self, last_state, last_action, state):
        return self.transition_gen_(self, last_state, last_action, state, self.transition_gen_params)
    
    def get_values(self):
        return self.values

    def set_values(self):
        self.values = np.dot(self.weights, self.all_state_vectors)

    def start(self, state, action, rho = 1.0):
        self.last_state = state
        self.last_action = action
        self.eligTrace = np.zeros(self.num_states)
        
        return action
    
    def step(self, state, rho = 1.0): 
        # get relevant feature
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]
        
        cumulant, gamma = self.transition_gen(self.last_state, None, state)
        self.cumulant = cumulant
        
        # retrieve the values
        self.last_state_value = np.sum(self.weights*last_state_vector)
        self.current_state_value = np.sum(self.weights*current_state_vector)
        
        # GTD lambda updates
        self.delta = self.cumulant + (gamma * self.current_state_value - self.last_state_value)
        self.eligTrace = rho*(self.lambd * self.last_gamma * self.eligTrace + last_state_vector)
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.dot(self.eligTrace, self.hWeights)*current_state_vector)
        self.hWeights += self.alpha_h * (self.delta*self.eligTrace - np.sum(self.hWeights*last_state_vector)*last_state_vector)

        self.last_gamma = gamma
        self.prediction = self.last_state_value
        
        self.last_state = state
        
    def end(self, state, rho = 1.0): 
        # get relevant feature
        # get relevant feature
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]

        cumulant, gamma = self.transition_gen(self.last_state, None, state)

        self.cumulant = cumulant
        
        # retrieve the values
        self.last_state_value = np.sum(self.weights*last_state_vector)
        self.current_state_value = np.sum(self.weights*current_state_vector)
        
        # GTD lambda updates
        self.delta = self.cumulant + (gamma * self.current_state_value - self.last_state_value)
        self.eligTrace = rho*(self.lambd * self.last_gamma * self.eligTrace + last_state_vector)
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.dot(self.eligTrace, self.hWeights)*current_state_vector)
        self.hWeights += self.alpha_h * (self.delta*self.eligTrace - np.sum(self.hWeights*last_state_vector)*last_state_vector)

        self.last_gamma = gamma
        self.prediction = self.last_state_value

        # update state values
        self.set_values()

        self.last_gamma = gamma
        self.prediction = self.last_state_value    