import numpy as np
import pandas as pd

class MonteCarloGVD:
    def __init__(self, agent_info={}):
        """Implements the SeparateActionGVD for a SeparateHorde.
            Args:
                agent_info (dictionary) : contains the GVD parameters
            Initialises:
                self.num_states (int): the dimension of the state space
                self.num_actions (int): the number of actions
                self.all_state_vectors (Numpy array): state vector representation of all states
                self.transition_gen (function): transition function (e.g. deterministic cumulant)
                self.transition_gen_params (dictionary): the parameters for the transition_gen function
                self.control (boolean): if True, the target policy is the greedy policy w.r.t the q-values
                self.policy (Numpy array): target policy
                self.cumulants (list): list of observed cumulants
                self.gammas (list): list of observed gammas
                self.last_states (list): list of observed last_states
                self.last_actions (list): list of observed last_actions
                self.states (list): list of observed states
                self.actions (list): list of observed actions
                self.rhos (list): list of observed rhos
                self.prod_rhos (list): list of observed product of rhos
                self.returns (list): list of observed returns
            """
        self.num_states = agent_info.get("num_states", 60)
        self.num_actions = agent_info.get("num_actions", 4)
        # model params
        self.last_state = None
        self.all_state_vectors = np.diag(np.ones(self.num_states))
        self.steps = 0

        # transition gen
        self.transition_gen_ = agent_info.get("transition_gen")
        self.transition_gen_params = agent_info.get("transition_gen_params", {})

        # keep track of the cumulants, gammas, states, actions and returns
        self.cumulants = []
        self.gammas = []
        self.last_states = []
        self.last_actions = []
        self.states = []
        self.actions = []
        self.rhos = []
        self.prod_rhos = []
        self.returns = []
        # policy
        self.policy = agent_info.get("policy")

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

    def store_transition(self, last_state, last_action, state, action, cumulant, gamma, rho):
        self.last_states.append(last_state)
        self.last_actions.append(last_action)
        self.states.append(state)
        self.actions.append(action)
        self.cumulants.append(cumulant)
        self.gammas.append(gamma)
        self.rhos.append(rho)

    def start(self, state, action):
        """GVD starts"""
        self.last_state = state
        self.last_action = action
        return action

    def setDataFrame(self):
        """Sets a pandas DataFrame of last_states, last_actions, returns and weights"""
        self.df = pd.DataFrame({"last_state": self.last_states,
                           "last_action": self.last_actions,
                           "return": self.returns,
                            "weight" : self.prod_rhos})

    def get_returns(self, last_state, last_action):
        """
        Args:
            last_state (int)
            last_action (int)
        Returns:
            state_action_returns (Numpy array): A vector containing the associated returns.
        """
        mask = (self.df["last_state"] == last_state) & (self.df["last_action"] == last_action)
        state_action_returns  = np.array(self.df.loc[mask, "return"])
        return state_action_returns

    def get_weight_returns(self, last_state, last_action):
        """
        Args:
            last_state (int)
            last_action (int)
        Returns:
            state_action_returns (Numpy array): A vector containing the associated returns.
            state_action_weights (Numpy array): A vector containing the associated weights.
        """
        mask = (self.df["last_state"] == last_state) & (self.df["last_action"] == last_action)
        state_action_returns = np.array(self.df.loc[mask, "return"])
        state_action_weights = np.array(self.df.loc[mask, "weight"])
        return state_action_returns, state_action_weights

    def set_returns(self):
        """
        Called on episode end.
        Extends the list of returns and prod_rhos
        Reset the list of rhos, gammas and cumulants.
            """
        # called on episode end
        n = len(self.cumulants)
        returns = np.zeros(n)
        prod_rhos = np.zeros(n)
        for j in range(n-1, -1, -1):
            if j == len(returns) - 1:
                returns[j] = self.cumulants[j]
                prod_rhos[j] = 1
            else:
                returns[j] = self.cumulants[j] + self.gammas[j]*returns[j+1]
                if self.gammas[j] == 0:
                    prod_rhos[j] = 1
                else:
                    prod_rhos[j] = self.rhos[j+1]*prod_rhos[j+1]

        # reset the cumulants, rhos, and gammas
        self.rhos = []
        self.gammas = []
        self.cumulants = []
        self.prod_rhos.extend(prod_rhos.tolist())
        self.returns.extend(returns.tolist())

    def step(self, state, action, rho):
        """GVD step"""
        # get relevant feature
        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)
        # store transition
        self.store_transition(self.last_state, self.last_action, state, action, cumulant, gamma, rho)

        self.last_state = state
        self.last_action = action
        self.steps += 1

    def end(self, state, action, rho):
        """GVD end"""
        # get relevant feature
        cumulant, gamma = self.transition_gen(self.last_state, self.last_action, state)

        # store transition
        self.store_transition(self.last_state, self.last_action, state, action, cumulant, gamma, rho)

        # set the returns
        self.set_returns()
        # put in the buffer
        self.steps += 1

