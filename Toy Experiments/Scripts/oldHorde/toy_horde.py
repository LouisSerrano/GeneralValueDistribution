### inherit the abstract class
import numpy as np
from Scripts.horde import BaseHorde
from Scripts.oldHorde.valueGVF import ValueGVF
from Scripts.oldHorde.actionValueGVF import ActionValueGVF
from Scripts.oldHorde.quantileGVD import QR_ActionGVF, ActionValueModel

class Horde(BaseHorde):
    """Implements the Horde for an RLGlue environment.
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
           self.on_policy (boolean): determines if the horde is off-policy or on-policy
           self.epsilon (float): Initial epsilon value. Parameter for epsilon-greedy policy
           self.epsilon_step (float): epsilon value at current step
           self.lambd (float): parameter for softmax policy
           self.epsilon_greedy (boolean): The behaviour policy is epsilon-greedy when True
           self.soft_max (boolean): The behaviour policy is soft-max when True
           self.on_policy_GVD (int): The index of the on policy GVD
           self.policy_update_freq (int): The frequency at which we update the behaviour policy
           self.valueGVF_number (int): The number of state value GVF
           self.actionGVF_number (int): The number of state-action value GVF
           self.QR_actionGVF_number (int): The number of QR-GVD
           self.GVFs (dictionary): contains the GVFs
           self.visitation_state (Numpy array): an array that keeps track of the visited states
           self.steps (int): counts the number of steps
           self.episodes (int): counts the number of episodes
       """

        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        self.rand_generator = np.random.RandomState(horde_info.get("seed"))
        # behaviour policy 
        self.policy = horde_info.get("policy")
        self.on_policy = horde_info.get("on_policy", False)
        if self.on_policy:
            self.epsilon = horde_info.get("epsilon", 0.05)
            self.lambd = horde_info.get("lambd", 1)
            self.epsilon_greedy = horde_info.get("epsilon_greedy", False)
            self.soft_max = horde_info.get('soft_max', False)
            self.on_policy_GVF = horde_info.get("on_policy_GVF")
            self.policy_update_freq = horde_info.get("policy_update_freq", 5)
        self.valueGVF_number = horde_info.get("valueGVF_number", 0)
        self.actionGVF_number = horde_info.get("actionGVF_number", 0)
        self.QR_actionGVF_number = horde_info.get("QR_actionGVF_number", 0)
        self.visitation_state = np.zeros(60)
        self.steps = 0
        
        dic = {}
        for j in range(1, self.valueGVF_number + 1):
            name = "V{}".format(j)
            if(horde_info.get(name) != None):
                dic[name] = ValueGVF(horde_info.get(name))
        for j in range(1, self.actionGVF_number + 1):
            name = "A{}".format(j)
            if(horde_info.get(name) != None):
                dic[name] = ActionValueGVF(horde_info.get(name))
        for j in range(1, self.QR_actionGVF_number + 1):
            name = "QR_A{}".format(j)
            if(horde_info.get(name) != None):
                dic[name] = QR_ActionGVF(horde_info.get(name))
        self.GVFs = dic
    
    def get_state_distribution(self):
        """
        Returns:
            mu (Numpy array): the state frequencies """
        return self.visitation_state / self.steps

    def horde_start(self, state):
        """ The first method called when the experiment starts, called after
        the environment starts.
        Runs sequentially the start method for every GVF.

        Args:
            state (int): the state observation from the environment's env_start function.
        Returns:
            The first action the horde takes.
        """
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

        # value gvf
        for j in range(1, self.valueGVF_number + 1):
            name = "V{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                gvf.start(state, action)
                self.GVFs.update({name : gvf})
            
        # action value gvf
        for j in range(1, self.actionGVF_number + 1):
            name = "A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                gvf.start(state, action)
                self.GVFs.update({name : gvf})
        
        # qr action gvf
        for j in range(1, self.QR_actionGVF_number + 1):
            name = "QR_A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                gvf.start(state, action)
                self.GVFs.update({name: gvf})
            
        self.last_state = state
        self.last_action = action
        self.visitation_state[state] += 1
        self.steps +=1
        
        return action

    def horde_step(self, state):
        """A step taken by the horde. The horde sequentially runs the step method for every GVF.
        Args:
            state (int): the state observation from the
                environment's step based, where the horde ended up after the
                last step
        Returns:
            The action the horde is taking.
        """
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
                
        # value gvf
        for j in range(1, self.valueGVF_number + 1):
            name = "V{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                rho = gvf.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
                gvf.step(state, rho)
                self.GVFs.update({name : gvf})
            
        # action value gvf    
        for j in range(1, self.actionGVF_number + 1):
            name = "A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                rho = gvf.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
                gvf.step(state, action, rho)
                self.GVFs.update({name : gvf})
                
        # qr action gvf
        for j in range(1, self.QR_actionGVF_number + 1):
            name = "QR_A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                gvf.step(state, action)
                self.GVFs.update({name: gvf})
        
        self.last_state = state
        self.last_action = action
        self.visitation_state[state] += 1
        self.steps +=1
        
        if self.on_policy:
            if self.steps % self.policy_update_freq == 0:
                gvf = self.GVFs[self.on_policy_GVF]
                if self.epsilon_greedy:
                    self.policy = gvf.update_epsilon_greedy_policy(self.epsilon)
                elif self.soft_max:
                    self.policy = gvf.update_soft_max_policy(self.lambd)
        
        return self.last_action

    def horde_end(self, state):
        """Run when the agent terminates. The horde sequentially runs the end method for every GVF.
        """
        
        ## the action does not matter actually
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        
        # value gvf
        for j in range(1, self.valueGVF_number + 1):
            name = "V{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                rho = gvf.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
                gvf.end(state, rho)
                self.GVFs.update({name : gvf})
            
        # action value gvf    
        for j in range(1, self.actionGVF_number + 1):
            name = "A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                rho = gvf.policy[self.last_state, self.last_action] / self.policy[self.last_state, self.last_action]
                gvf.end(state, action, rho)
                self.GVFs.update({name : gvf})
                
        # qr action gvf
        for j in range(1, self.QR_actionGVF_number + 1):
            name = "QR_A{}".format(j)
            gvf = self.GVFs.get(name)
            if gvf != None:
                gvf.end(state, action)
                self.GVFs.update({name: gvf})
                
        self.visitation_state[state] += 1
        self.steps +=1
        
        
        self.visitation_state[state] += 1
        self.steps +=1
        if self.on_policy:
            gvf = self.GVFs[self.on_policy_GVF]
            if self.epsilon_greedy:
                self.policy = gvf.update_epsilon_greedy_policy(self.epsilon)
            elif self.soft_max:
                self.policy = gvf.update_soft_max_policy(self.lambd)
            
    def horde_cleanup(self):
        """Cleanup done after the agent ends."""

    def horde_message(self, message, GVF = "V1"):
        if message == "get state values":
            return self.GVFs[GVF].get_values()
        if message == "get state distribution":
            return self.visitation_state/self.steps
        if message == "get action values":
            return self.GVFs[GVF].get_q_values()
