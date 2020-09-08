from Scripts.environment_horde import BaseEnvironmentHorde
import numpy as np


class ToyEnvironment(BaseEnvironmentHorde):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):

        self.maze_dim = [6,10]
        self.obstacles = [[2,1],[3,1],[4,1],[0,4],[1,4],[5,5],[1,7],[2,7],[2,8],[3,8]]

        self.start_state = [5,0]
        self.current_state = [None, None]

        observation = None
        termination = None
        self.obs_term = [observation, termination]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the first state, boolean
            indicating if it's terminal.
        """

        self.obs_term = [None, False]
        self.end_states = env_info.get("end_states", [[0,8]])

    def env_start(self):
        """The first method called when the experiment starts, called before the
        horde starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.obs_term[0] = self.get_observation(self.current_state)

        return self.obs_term[0]

    # check if current state is within the gridworld and return bool
    def out_of_bounds(self, row, col):
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    # check if there is an obstacle at (row, col)
    def is_obstacle(self, row, col):
        if [row, col] in self.obstacles:
            return True
        else:
            return False

    def get_observation(self, state):
        """
        Args:
            state: A state represented by a 2D-array [s_i, s_j]
        Returns:
            (observation_state): The state observation is a numpy array representing the state.
             We represent [s_i, s_j] by s_i * self.maze_dim[1] + s_j.
             Throughout the project, we generally make no distinction between observation and state
              and call them both state unless when required.
        """
        return state[0] * self.maze_dim[1] + state[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the horde

        Returns:
            (state, Boolean): a tuple of the state observation,
                and boolean indicating if it's terminal.
        """
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # update current_state with the action (also check validity of action)
        if action == 0: # up
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        elif action == 1: # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2: # down
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3: # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

        if self.current_state in self.end_states: # terminate if goal is reached
            is_terminal = True

        self.obs_term = [self.get_observation(self.current_state), is_terminal]

        return self.obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            Object: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.obs_term[0])

        # else
        return "I don't know how to respond to your message"