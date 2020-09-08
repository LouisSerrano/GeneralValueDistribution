
"""Glues together an experiment, horde, and environment.
"""

from __future__ import print_function


class RLGlueHorde:
    """RLGlueHorde class

    Args:
        env_name (string): the name of the module where the Environment class can be found
        horde_name (string): the name of the module where the Horde class can be found
    """

    def __init__(self, env_class, horde_class):
        self.environment = env_class()
        self.horde = horde_class()

        #self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self, horde_init_info={}, env_init_info={}):
        """Initial method called when RLGlueHorde experiment is created"""
        self.environment.env_init(env_init_info)
        self.horde.horde_init(horde_init_info)

        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self):
        """Starts RLGlueHorde experiment

        Returns:
            tuple: (state, action)
        """
        
        self.num_steps = 1

        last_state = self.environment.env_start()
        self.last_action = self.horde.horde_start(last_state)

        last_state_action = (last_state, self.last_action)

        return last_state_action

    def rl_horde_start(self, observation):
        """Starts the horde.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the horde.
        """
        return self.horde.horde_start(observation)

    def rl_horde_step(self, state):
        """Step taken by the horde

        Args:
            state (Numpy array): the current state of the horde in the environment

        Returns:
            The action taken by the horde in the current state
        """
        return self.horde.horde_step(state)

    def rl_horde_end(self, state):
        """Run when the horde terminates

        Args:
            state (Numpy array): the current state of the horde when terminating
        """
        self.horde.horde_end(state)

    def rl_env_start(self):
        """Starts RLGlueHorde environment.

        Returns:
            (state, Boolean): state observation, boolean
                indicating termination
        """
        self.num_steps = 1
        this_observation = self.environment.env_start()

        return this_observation

    def rl_step(self):
        """Step taken by RLGlueHorde, takes environment step and either step or
            end by horde.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        # same, env_step should only return a last_state and terminal
        (last_state, term) = self.environment.env_step(self.last_action)

        if term:
            self.num_episodes += 1
            self.horde.horde_end(last_state)
            roat = (last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.horde.horde_step(last_state)
            roat = (last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.horde.horde_cleanup()

    def rl_horde_message(self, message):
        """Message passed to communicate with horde during experiment

        Args:
            message: the message (or question) to send to the horde

        Returns:
            The message back (or answer) from the horde

        """
        return self.horde.horde_message(message, )

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode
            If max_steps_this_episode == 0, there is no step limit.

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[2]

        return is_terminal

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes
        """
        return self.num_episodes
