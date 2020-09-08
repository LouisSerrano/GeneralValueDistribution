from Scripts.utils import MazeTileCoder
import numpy as np


class ValueGVF:
    def __init__(self, agent_info={}):
        self.lambd = agent_info.get("lambda", 0.1)
        self.alpha = agent_info.get("alpha", 0.1)
        self.beta =  0.1*self.alpha
        self.gamma = agent_info.get("gamma", 0.95)
        self.last_gamma = self.gamma
        self.current_gamma = None
        self.last_state = None
        
        self.num_tilings = agent_info.get("num_tilings", 4)
        self.num_tiles = agent_info.get("num_tiles", 4)
        self.tc = MazeTileCoder(num_tilings=self.num_tilings, 
                                         num_tiles=self.num_tiles)
        self.iht_size = self.tc.iht_size
        self.weights = np.zeros(self.iht_size)
        self.hWeights = np.zeros(self.iht_size)
        self.eligTrace = np.zeros(self.iht_size)

        self.objective_states = agent_info.get("objective_states", [[2,4]])
        if len(self.objective_states) == 1:
            self.objective_states = self.objective_states[0]
            
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})
        
        self.policy = agent_info.get("policy")
        self.values = np.zeros((6,10))
        obstacles = [[2,1],[3,1],[4,1],[0,4],[1,4],[5,5],[1,7],[2,7],[2,8],[3,8]]
        for (i,j) in obstacles:
            self.values[i,j] = np.nan
        
    def transition_gen(self, state):
        return self.transition_gen_(self.objective_states, state, self.gamma, self.transition_gen_params)
    
    def get_values(self):
        return self.values

    def start(self, state, action, rho = 1.0):
        self.last_state = state
        self.last_action = action
        self.eligTrace = np.zeros(self.iht_size)
        
        return action
    
    def step(self, state, rho = 1.0): 
        # get relevant feature
        last_i, last_j = self.tc.get_observation(self.last_state)
        last_state_vector = self.tc.get_state_vector(self.last_state)
        current_state_vector = self.tc.get_state_vector(state)
        
        cumulant, gamma = self.transition_gen(state)
        self.cumulant = cumulant
        
        ## retrieve the values
        self.last_state_value = np.sum(self.weights*last_state_vector)
        self.current_state_value = np.sum(self.weights*current_state_vector)
        
        ##GTD lambda updates
        self.delta = self.cumulant + (gamma * self.current_state_value - self.last_state_value)
        self.eligTrace = rho*(self.lambd * self.last_gamma * self.eligTrace + last_state_vector)
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.dot(self.eligTrace, self.hWeights)*current_state_vector)
        self.hWeights += self.beta * (self.delta*self.eligTrace - np.sum(self.hWeights*last_state_vector)*last_state_vector)
        
        self.values[last_i, last_j] = np.sum(self.weights*last_state_vector)
        
        self.last_gamma = gamma
        self.prediction = self.last_state_value
        
        self.last_state = state
        
    def end(self, state, rho = 1.0): 
        # get relevant feature
        last_i, last_j = self.tc.get_observation(self.last_state)
        last_state_vector = self.tc.get_state_vector(self.last_state)
        current_state_vector = self.tc.get_state_vector(state)
        
        cumulant, gamma = self.transition_gen(state)
        self.cumulant = cumulant
        
        ## retrieve the values
        self.last_state_value = np.sum(self.weights*last_state_vector)
        self.current_state_value = np.sum(self.weights*current_state_vector)
        
        ##GTD lambda updates
        self.delta = self.cumulant + (gamma * self.current_state_value - self.last_state_value)
        self.eligTrace = rho*(self.lambd * self.last_gamma * self.eligTrace + last_state_vector)
        self.weights += self.alpha *(self.delta * self.eligTrace - gamma*(1-self.lambd)*np.dot(self.eligTrace, self.hWeights)*current_state_vector)
        self.hWeights += self.beta * (self.delta*self.eligTrace - np.sum(self.hWeights*last_state_vector)*last_state_vector)
        
        self.values[last_i, last_j] = np.sum(self.weights*last_state_vector)
        
        self.last_gamma = gamma
        self.prediction = self.last_state_value
        
        
        self.last_gamma = gamma
        self.prediction = self.last_state_value    