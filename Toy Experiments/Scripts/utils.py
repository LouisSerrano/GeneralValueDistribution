import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import scipy.stats as st
import Scripts.oldHorde.tiles3 as tc
from collections import deque
import random
import datetime

# utils is a script file that contains useful functions for the project

# 1. Plotting functions

#plot script for rmsve

def plot_script(filename = None):
    tilings = []
    tiles = []
    alphas = []
    RMSVEs = []
    if filename == None:
        RMSVE_names = [name for name in os.listdir() if name.startswith("RMSVE")]
    else:
        RMSVE_names = [name for name in os.listdir(filename) if name.startswith("RMSVE")]
    for name in RMSVE_names:
        l = name.split('_')
        tiling = int(l[4])
        tile = int(l[6])
        s = l[8].split('.')[0]
        alpha = int(s)/10**(len(s)-1)
        tilings.append(tiling)
        tiles.append(tile)
        alphas.append(alpha)
        if filename == None:
            RMSVEs.append(np.load(name))
        else:
            RMSVEs.append(np.load(os.path.join(filename, name)))
    fig, ax = plt.subplots(figsize = (16,10))
    x = np.arange(len(RMSVEs[0]))
    for j in range(len(RMSVEs)):
        RMSVE = RMSVEs[j]
        ax.plot(x, RMSVE, label = "tilings : {}, tiles : {}, alpha: {}".format(tilings[j], tiles[j], alphas[j]))
        
    leg = ax.legend()

# plot quantiles
def plot_theta(theta, s, a, filename = None):
    """
      Args:
          theta (Numpy array) : Contains the quantiles for all state-action pairs
          s (int) : last state
          a (int) : last action
          filename (String) : where to store the plot

      Returns:
          A cdf plot of the return distribution at (s,a)
        """
    t = theta[s, a]
    n = t.shape[0]
    tau = (2 * np.arange(n) + 1) / (2.0 * n)
    plt.figure(figsize = (8,6))
    plt.plot(t, tau, marker = 'o', linestyle = "--", label = "state : {}, action : {}".format(s,a))
    plt.legend()
    plt.ylabel("quantile levels " r'$\tau$')
    plt.xlabel("$F^{-1}$("r'$\tau$'")")
    if filename is not None:
        plt.savefig(filename)

# plot state values and policy
def visualize(values, policy, experiment_name=''):
    """
      Args:
          values (Numpy array) : state values
          policy (Numpy array) : policy
          filename (String) : where to store the plot

      Returns:
          A heatmap plot with arrows representing the policy and colours accounting for the state values
    """
    grid_h = 6
    grid_w = 10
    fig = plt.figure(figsize=(10, 12))
    cmap = matplotlib.cm.Blues
    fig.clear()

    plt.xticks([])
    plt.yticks([])
    im = plt.imshow(values, cmap=cmap, interpolation='nearest', origin='upper')

    for state in range(policy.shape[0]):
        for action in range(policy.shape[1]):
            y, x = np.unravel_index(state, (grid_h, grid_w))
            pi = policy[state][action]
            if pi == 0:
                continue
            if action == 0:
                plt.arrow(x, y, 0, -0.3 * pi, fill=False, length_includes_head=False, head_width=0.08,
                          alpha=0.5)
            if action == 1:
                plt.arrow(x, y, 0.3 * pi, 0, fill=False, length_includes_head=False, head_width=0.08,
                          alpha=0.5)
            if action == 2:
                plt.arrow(x, y, 0, +0.3 * pi, fill=False, length_includes_head=False, head_width=0.08,
                          alpha=0.5)
            if action == 3:
                plt.arrow(x, y, -0.3 * pi, 0, fill=False, length_includes_head=False, head_width=0.08,
                          alpha=0.5)

    plt.title((("" or experiment_name) + "\n"))
    plt.colorbar(im, orientation='vertical', fraction=0.03, pad=0.04)

# plot q_values heatmap
def plot_q_values(q):
    """
      Args:
          q (Numpy array) : state-action values

      Returns:
          A heatmap plot with q(s, up), q(s, right), q(s, left), q(s, down) for all states s
        """
    fig = plt.figure(figsize =(16,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    sns.heatmap(q[:, 0].reshape(6,10), ax = ax1, cmap = "Blues")
    sns.heatmap(q[:, 1].reshape(6,10), ax = ax2, cmap = "Blues")
    sns.heatmap(q[:, 2].reshape(6,10), ax = ax3, cmap = "Blues")
    sns.heatmap(q[:, 3].reshape(6,10), ax = ax4, cmap = "Blues")
    
    ax1.set_title("Up action")
    ax2.set_title("Right action")
    ax3.set_title("Down action")
    ax4.set_title("Left action")

# save theta values
def save_theta(path, gvd_params, exp_params, theta):
    """
      Args:
          path (String) : where to save theta
          gvd_params (dictionary) : parameters of the GVD
          exp_params (dictionary) : parameters of the experiment
          theta (Numpy array) : quantiles for all state-action pairs
        """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(path, current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + "/theta", theta)
    with open(path + "/gvd_params.txt", 'w') as f:
        print(gvd_params, file=f)
    with open(path + "/exp_params.txt", 'w') as f:
        print(exp_params, file=f)


# 2. Maze tile coder used to represent states of the maze
    
# maze tile coder for the environment
class MazeTileCoder:
    def __init__(self, num_tilings=4, num_tiles=4):
        """
        Initializes the Maze Tile Coder
        Initializers:
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht_size -- the size of the index hash table, computed with num_tilings*(num_tiles+1)*(num_tiles+1)
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht_size = num_tilings*(num_tiles+1)*(num_tiles+1)
        self.iht = tc.IHT(self.iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        
    def get_state(self, observation):
        """
        Args:
            observation (Numpy array): The state observation of the horde

        Returns:
            state (s_i, s_j) : the coordinates of the state in the maze environment
        """
        return observation//10, observation%10
    
    def get_tiles(self, observation):
        """
        Args:
            observation (Numpy array): The state observation of the horde
        Returns:
            tiles (Numpy array) : the active tiles of the state
        """
        i_scale = self.num_tiles / 5
        j_scale = self.num_tiles / 9

        position_i, position_j = self.get_state(observation)
        tiles = tc.tiles(self.iht, self.num_tilings, [(5 - position_i)*i_scale,
                                                      position_j*j_scale])
        return np.array(tiles)

    def get_state_vector(self, observation):
        """
        Args:
            observation (Numpy array): The state observation of the horde
        Returns:
            tile_vector (Numpy array) : A vector tile representation of the state.
            Non-zero entries of the vector are located at the index of the corresponding active tiles
        """
        active_tile = self.get_tiles(observation)
        vector = np.zeros(self.iht_size)
        vector[active_tile] = 1
        return vector

# 3. Experience Replay Buffer used for Quantile GVD

class ReplayBuffer:
    """
    Initializes the Replay Buffer used for training Quantile GVD
    Initializers:
        batch_size -- int, the size of the batch_size
        capacity -- int, the maximum capacity of the Deque. Once the maximum size is reached,
        the oldest elements of the Buffer are pushed away by the most recent
    Class Variables:
        self.buffer -- deque(capacity)
        self.batch_size -- batch_size used for sampling
    """

    def __init__(self, batch_size = 32, capacity=10000000):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, last_state_vector, last_action, cumulant, gamma, current_state, current_state_vector, next_action):
        """
        Stores transitions into the replay buffer
        Args :
            last_state_vector (Numpy array) : vector representation of the last state
            last_action (Numpy array) : one-hot encoded representation of the last action
            cumulant (float) : observed cumulant when transitioning from (last_action, last_state) to (current_state)
            gamma (float) : gamma of the current state
            current_state (int) : observation of the current state
            current_state_vector (Numpy array) : vector representation of the current state
            next_action (int) : the next_action taken by the horde
            """
        self.buffer.append([last_state_vector, last_action, cumulant, gamma, current_state, current_state_vector, next_action])

    def sample(self):
        """
        Samples transitions from the dequeu
        Returns :
            last_state_vectors (Numpy array) : (batch_size, 60)
            last_actions (Numpy array) : (batch_size, 4)
            cumulants (Numpy array) : (batch_size)
            gammas (Numpy array) : (batch_size)
            current_states (Numpy array) : (batch_size)
            current_state_vectors (Numpy array) : (batch_size, 60)
            next_actions (Numpy array) : (batch_size)

        """
        sample = random.sample(self.buffer, self.batch_size)
        last_state_vectors, last_actions, cumulants, gammas, current_states, current_state_vectors, next_actions = map(
            np.asarray, zip(*sample))
        last_state_vectors = last_state_vectors.reshape(self.batch_size, -1)
        current_state_vectors = current_state_vectors.reshape(self.batch_size, -1)
        return last_state_vectors, last_actions, cumulants, gammas, current_states, current_state_vectors, next_actions

    def size(self):
        return len(self.buffer)


# 4. Cumulant functions

def deterministic_cumulant(gvd, last_obs, last_action, obs, params):
    """
    Args:
        gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
        last_obs (int) : Observation of the last state
        last_action (int) : last action taken bu the horde
        obs (int) : Observation of the current state
        params (dictionary) : Contains parameters for the cumulant and termination function

    Returns:
        cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
        The cumulant is deterministic.
        gamma (float) : termination signal of the current state
    """

    state = [obs // 10, obs % 10]
    r = params.get("r", 10)
    objective_states = params.get("objective_states", [[2,4]])
    gamma = params.get("gamma", 0.95)
    if state in objective_states:
        cumulant = r
        gamma = 0
    else:
        cumulant = 0
        gamma = gamma
    return cumulant, gamma


def bernouilli_cumulant(gvd, last_obs, last_action, obs, params):
    """
    Args:
        gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
        last_obs (int) : Observation of the last state
        last_action (int) : last action taken bu the horde
        obs (int) : Observation of the current state
        params (dictionary) : Contains parameters for the cumulant and termination function

    Returns:
        cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
        The cumulant is Bernouilli.
        gamma (float) : termination signal of the current state
        """

    state = [obs // 10, obs % 10]
    r = params.get("r", 10)
    objective_states = params.get("objective_states", [[4,8]])
    gamma = params.get("gamma", 0.95)
    if state in objective_states:
        cumulant = np.random.binomial(1, 0.5)*2*r
        gamma = 0
    else:
        cumulant = 0
        gamma = gamma
    return cumulant, gamma

def deterministic_vs_bernouilli(gvd, last_obs, last_action, obs, params):
    """
   Args:
       gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
       last_obs (int) : Observation of the last state
       last_action (int) : last action taken bu the horde
       obs (int) : Observation of the current state
       params (dictionary) : Contains parameters for the cumulant and termination function

   Returns:
       cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
       The cumulant is deterministic for the first objective state and Bernouilli for the second.
       gamma (float) : termination signal of the current state
       """
    state = [obs // 10, obs % 10]
    r = params.get("r", 10)
    objective_states = params.get("objective_states", [[2,4],[4,8]])
    gamma = params.get("gamma", 0.95)
    if state == objective_states[0]:
        cumulant = r
        gamma = 0
    elif state == objective_states[1]:
        cumulant = np.random.binomial(1, 0.5)*2*r
        gamma = 0
    else:
        cumulant = 0
    return cumulant, gamma


def gaussian_cumulant(gvd, last_obs, last_action, obs, params):
    """
   Args:
       gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
       last_obs (int) : Observation of the last state
       last_action (int) : last action taken bu the horde
       obs (int) : Observation of the current state
       params (dictionary) : Contains parameters for the cumulant and termination function

   Returns:
       cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
       The cumulant is Gaussian.
       gamma (float) : termination signal of the current state
       """
    mu = params.get('mu', 10)
    sigma = params.get('sigma', 2)
    state = [obs // 10, obs % 10]
    objective_states = params.get("objective_states", [[2,4]])
    gamma = params.get("gamma", 0.95)

    if state in objective_states:
        cumulant = st.norm(mu, sigma).rvs(1)
        gamma = 0
    else:
        cumulant = 0
        gamma = gamma
    return cumulant, gamma


def gaussian_vs_gaussian(gvd, last_obs, last_action, obs, params):
    """
      Args:
          gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
          last_obs (int) : Observation of the last state
          last_action (int) : last action taken bu the horde
          obs (int) : Observation of the current state
          params (dictionary) : Contains parameters for the cumulant and termination function

      Returns:
          cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
          The cumulant is Gaussian with std = sigma_1  for the first objective state and Gaussian with std = sigma_2
          for the second.
          gamma (float) : termination signal of the current state
          """
    mu = params.get('mu', 10)
    sigma_1 = params.get('sigma_1', 2)
    sigma_2 = params.get('sigma_2', 5)
    state = [obs // 10, obs % 10]
    objective_states = params.get("objective_states", [[2,4],[4,6]])
    gamma = params.get("gamma", 0.95)

    if state == objective_states[0]:
        cumulant = st.norm(mu, sigma_1).rvs(1)
        gamma = 0
    elif state == objective_states[1]:
        cumulant = st.norm(mu, sigma_2).rvs(1)
        gamma = 0
    else:
        cumulant = 0
        gamma = gamma
    return cumulant, gamma


def mixture_of_gaussians(gvd, last_obs, last_action, obs, params):
    """
  Args:
      gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
      last_obs (int) : Observation of the last state
      last_action (int) : last action taken bu the horde
      obs (int) : Observation of the current state
      params (dictionary) : Contains parameters for the cumulant and termination function

  Returns:
      cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
      The cumulant is a mixture of gaussian with a fixed mean of 10.
      gamma (float) : termination signal of the current state
    """

    pi_1 = params.get("pi_1", 0.6)
    mean = params.get("mean", 10)
    mu_1 = params.get('mu_1', 8)
    mu_2 = (mean - pi_1*mu_1)/(1-pi_1)
    sigma_1 = params.get("sigma_1", 2)
    sigma_2 = params.get("sigma_2", 5)
    state = [obs // 10, obs % 10]
    gamma = params.get("gamma", 0.95)
    objective_states = params.get("objective_states", [[2,4],[4,8]])

    def draw_sample(pi_1, mu_1, mu_2, sigma_1, sigma_2):
        if st.uniform().rvs(1) < pi_1:
            sample = st.norm(mu_1, sigma_1).rvs(1)
        else :
            sample = st.norm(mu_2, sigma_2).rvs(1)
        return sample

    if state in objective_states:
        cumulant = draw_sample(pi_1, mu_1, mu_2, sigma_1, sigma_2)
        gamma = 0
    else:
        cumulant = 0
        gamma = gamma
    return cumulant, gamma


def gvd_based_cumulant(gvd, last_obs, last_action, obs, params):
    """
      Args:
          gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
          last_obs (int) : Observation of the last state
          last_action (int) : last action taken bu the horde
          obs (int) : Observation of the current state
          params (dictionary) : Contains parameters for the cumulant and termination function

      Returns:
          cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
          The cumulant is the maximum of the q-values of input gvd at the current state.
          gamma (float) : termination signal of the current state
        """
    index = gvd.input - 1
    model = gvd.horde_z
    # max mean of the next state_action_pair
    state_vector = np.reshape(gvd.all_state_vectors[obs], (1, -1))
    thetas = model.gvd_model(state_vector , index)[0]
    cumulant = np.max(np.mean(thetas, -1))

    return cumulant, 0

def wall_detector_cumulant(gvd, last_obs, last_action, obs, params):
    """
      Args:
          gvd (GVD-like object) : A GVD Object. It is used when defining cumulants based on theta values of a particular GVD
          last_obs (int) : Observation of the last state
          last_action (int) : last action taken bu the horde
          obs (int) : Observation of the current state
          params (dictionary) : Contains parameters for the cumulant and termination function

      Returns:
          cumulant (float) : cumulant observed when transitioning from (last state, last action) to (current state).
          The is Bernouilli when in motion and the current state is near a wall.
          gamma (float) : termination signal of the current state
            """
    near_wall_states = params.get("near_wall_states")
    p = params.get("p", 0.8)
    cumulant = 0
    gamma = 0.1
    state = [obs // 10, obs % 10]
    last_state = [last_obs // 10, last_obs % 10]
    if last_obs != obs:
        if obs in near_wall_states:
            cumulant = st.bernoulli(p).rvs(1)
            gamma = params.get('gamma', 0.95)
    return cumulant, gamma