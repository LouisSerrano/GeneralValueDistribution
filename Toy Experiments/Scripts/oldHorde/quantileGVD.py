from Scripts.utils import ReplayBuffer
import numpy as np
import tensorflow as tf


class QR_ActionGVF:
    def __init__(self, agent_info={}):
        self.gamma = agent_info.get("gamma", 0.95)
        self.num_states = agent_info.get("num_states", 60)
        self.num_actions = agent_info.get("num_actions", 4)
        self.last_gamma = self.gamma
        self.beta = agent_info.get("beta_risk", 0)
        self.current_gamma = None
        self.last_state = None
        self.num_tilings = agent_info.get("num_tilings", 4)
        self.num_tiles = agent_info.get("num_tiles", 4)
        #self.tc = MazeTileCoder(num_tilings=self.num_tilings, 
        #                                 num_tiles=self.num_tiles)
        #self.iht_size = self.tc.iht_size
        self.iht_size = self.num_states
        ## model param
        self.num_quantiles = agent_info.get("num_quantiles", 51)
        self.update_freq = agent_info.get("update_freq", 1)
        self.tau = tf.constant((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), dtype = tf.float32)
        self.kappa = agent_info.get("kappa", 1)
        self.eta = agent_info.get("eta", 1e-3)
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        
        #for s in range(60):
        #    self.all_state_vectors[s, :] = self.tc.get_state_vector(s)
        
        self.q = ActionValueModel(self.iht_size,
                          self.num_actions, self.num_quantiles, self.kappa)
        self.q_target = ActionValueModel(self.iht_size,
                          self.num_actions, self.num_quantiles, self.kappa)
        self.batch_size = agent_info.get("batch_size", 32)
        self.per_step = agent_info.get("per_step", False)
        
        ## transition gen
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})
        
        ## environment
        self.objective_states = agent_info.get("objective_states", [[2,4]])
        if len(self.objective_states) == 1:
            self.objective_states = self.objective_states[0]
        ## keep track of the cumulants, gammas, states, actions and returns
        self.buffer = ReplayBuffer(self.batch_size)
        self.steps = 0
        
        ## policy
        self.control = agent_info.get("control", False)
        if self.control:
            self.policy = np.ones((self.num_states, self.num_actions))/self.num_actions
        if not self.control:
            self.policy = agent_info.get("policy")
        
    def transition_gen(self, state):
        return self.transition_gen_(self.objective_states, state, self.gamma, self.transition_gen_params)
    
    
    def set_q_values(self):
        quantiles = self.q.predict(self.all_state_vectors)
        q_values = np.mean(quantiles, -1)
        self.q_values = q_values
    
    def get_q_values(self):
        self.set_q_values()
        return self.q_values
    
    def set_q_beta(self):
        quantiles = self.q.predict(self.all_state_vectors)
        q_beta = 1/self.beta*np.log(np.mean(np.exp(quantiles*self.beta), -1))
        self.q_beta = q_beta
        
    def get_q_beta(self):
        self.set_q_beta()
        return self.q_beta
    
    def get_values(self):
        self.set_q_values()
        if self.control:
            self.set_policy()
        return np.sum(self.q_values*self.policy, -1)

    def set_policy(self):
        policy = np.zeros((self.num_states, self.num_actions))
        if self.beta == 0:
            q_values = self.get_q_values()
        else: 
            q_values = self.get_q_beta()
        for s in range(self.num_states):
            policy[s, :] = 0
            arg = self.argmax(self.q_values[s])
            policy[s, arg] = 1/len(arg)
        self.policy = policy
    
    def get_policy(self):
        self.set_policy()
        return self.policy
    
    def argmax(self, q_values):
        arg = np.argwhere(q_values == np.amax(q_values))
        return arg.flatten().tolist()
    
    def update_soft_max_policy(self, lambd):
        if self.beta == 0:
            q_values = self.get_q_values()
        else: 
            q_values = self.get_q_beta()
        m = np.max(lambd*q_values, -1, keepdims = True)
        e = np.exp(lambd*q_values - m)
        e_sum = np.expand_dims(np.sum(e, -1), -1)
        policy = e/ e_sum
        
        return policy
    
    
    def update_epsilon_greedy_policy(self, epsilon):
        policy = np.zeros((self.num_states, self.num_actions))
        if self.beta == 0:
            q_values = self.get_q_values()
        else: 
            q_values = self.get_q_beta()
        for s in range(self.num_states):
            arg = self.argmax(q_values[s])
            for a in range(self.num_actions):
                if a in arg:
                    policy[s, a] = (1-epsilon)/len(arg) + epsilon/self.num_actions
                else:
                    policy[s, a] = epsilon/self.num_actions
        return policy
        
    
    def replay(self):
        vector_states, actions, rewards, gammas, next_states, next_vector_states, current_actions = self.buffer.sample()
        q = self.q_target.predict(next_vector_states)
        theta = np.zeros((self.batch_size, self.num_quantiles))
        for i in range(self.batch_size):
            if self.control:
                if self.beta == 0:
                    current_actions = np.argmax(np.mean(q, axis=2), axis=1)
                else:
                    current_actions = np.argmax(1/self.beta*np.log(np.mean(np.exp(q*self.beta), axis = 2)), axis = 1)        
                theta[i,:] = rewards[i] + gammas[i] * q[i][current_actions[i]]
            else:
                pi = np.expand_dims(self.policy[next_states[i]], -1)
                theta[i,:] = rewards[i] + gammas[i] * np.sum(q[i]*pi, 0)
            
        self.q.train(vector_states, theta, actions)
        
    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)
    
    def start(self, state, action):
        self.last_state = state
        self.last_action = action
        
        return action
    
    def step(self, state, action): 
        # get relevant feature
        #last_state_vector = self.tc.get_state_vector(self.last_state)
        #current_state_vector = self.tc.get_state_vector(state)
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]
        
        cumulant, gamma = self.transition_gen(state)
        action_ = np.zeros(self.num_actions, dtype = np.float32)
        action_[self.last_action] = 1
        
        ## put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)
        ## train
        if self.per_step:
            last_state_vector = np.array(last_state_vector).reshape(1, -1)
            current_state_vector = np.array(current_state_vector).reshape(1, -1)
            
            q = self.q_target.predict(current_state_vector)
            theta = np.zeros((1, self.num_quantiles))

            if self.control:
                if self.beta == 0:
                    current_action = np.argmax(np.mean(q, axis=2), axis=1)
                else:
                    current_action = np.argmax(1/self.beta*np.log(np.mean(np.exp(q*self.beta), axis = 2)), axis = 1)
                theta[0,:] = cumulant + gamma * q[0][current_action]
                
            else: 
                pi = np.expand_dims(self.policy[state], -1)
                theta[0,:] = cumulant + gamma * np.sum(q[0]*pi, 0)
            
            self.q.train(last_state_vector, theta, action_)
            
        else:
            if self.buffer.size() > 50:
                self.replay()
                
        if self.steps % self.update_freq == 0:
            self.target_update()
        
        self.last_gamma = gamma
        self.last_state = state
        self.last_action = action
        self.steps += 1
        
        
    def end(self, state, action): 
        # get relevant feature
        #last_state_vector = self.tc.get_state_vector(self.last_state)
        #current_state_vector = self.tc.get_state_vector(state)
        last_state_vector = self.all_state_vectors[self.last_state]
        current_state_vector = self.all_state_vectors[state]
        
        cumulant, gamma = self.transition_gen(state)
        action_ = np.zeros(self.num_actions, np.float32)
        action_[self.last_action] = 1.0
        
        ## put in the buffer
        self.buffer.put(last_state_vector, action_, cumulant, gamma, state, current_state_vector, action)
        ## train
        if self.per_step:
            last_state_vector = np.array(last_state_vector).reshape(1, -1)
            current_state_vector = np.array(current_state_vector).reshape(1, -1)
            
            q = self.q_target.predict(current_state_vector)
            theta = np.zeros((1, self.num_quantiles))

            if self.control:
                if self.beta == 0:
                    current_action = np.argmax(np.mean(q, axis=2), axis=1)
                else:
                    current_action = np.argmax(1/self.beta*np.log(np.mean(np.exp(q*self.beta), axis = 2)), axis = 1)
                    theta[0,:] = cumulant + gamma * q[0][current_action]
                    
            else: 
                pi = np.expand_dims(self.policy[state], -1)
                theta[0,:] = cumulant + gamma * np.sum(q[0]*pi, 0)
            
            self.q.train(last_state_vector, theta, action_)
        
        else:
            if self.buffer.size() > 50:
                self.replay()
        if self.steps % 1 == 0:
            self.target_update()
            if self.control:
                self.set_policy()
        
        self.last_gamma = gamma
        self.steps += 1
        #self.last_state = state
        #self.last_action = action 
        
        
    
class ActionValueModel:
    def __init__(self, state_dim, num_actions = 4, num_quantiles = 51, kappa = 1, eta = 1e-3):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        self.eta = eta
        self.tau = tf.constant((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), dtype = tf.float32)
        #self.huber_loss = tf.keras.losses.Huber(
           # reduction=tf.keras.losses.Reduction.NONE)
        self.opt = tf.keras.optimizers.Adam(self.eta)
        self.model = self.create_model()
        self.backlog = {}
        self.backlog["losses"] = []
            
    def create_model(self):
        return tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=[self.state_dim, ]),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(self.num_actions*self.num_quantiles),
        tf.keras.layers.Reshape((self.num_actions, self.num_quantiles))])
    
    @tf.function
    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.expand_dims(pred, axis = 0)
        target_tile = tf.expand_dims(tf.transpose(target), axis=-1)
        target_tile = tf.cast(target_tile, tf.float32)
        u = target_tile - pred
        huber_loss = tf.where(tf.math.abs(u) < self.kappa,
                             1/2*tf.math.pow(u, 2),
                             self.kappa*(tf.math.abs(u) - 1/2*self.kappa))
        tau = tf.reshape(self.tau, [1, self.num_quantiles])
        
        loss = huber_loss*tf.math.abs(self.tau - tf.where(u < 0, 1.0, 0.0))
        loss = tf.transpose(loss, [1, 0,2])
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=1), axis=-1))
        return loss
    
    @tf.function
    def train(self, states, target, actions):
        with tf.GradientTape() as tape:
            theta = self.model(states)
            loss = self.quantile_huber_loss(target, theta, actions)
            self.backlog["losses"].append(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        
    def predict(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, self.state_dim)
        return self.model.predict(state)
    
    """def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)"""

    def get_optimal_action(self, state):
        z = self.model.predict(state)[0]
        q = np.mean(z, axis=1)
        return np.argmax(q)