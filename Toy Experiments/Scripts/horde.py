"""An abstract class that specifies the Horde API for rl_glue_horde.py.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod

class BaseHorde:
    """Implements the horde for an RL-Glue-Horde environment.
    Note:
        horde_init, horde_start, horde_step, horde_end, horde_cleanup, and
        horde_meassage are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def horde_init(self, horde_info= {}):
        """Setup for the horde called when the experiment first starts."""

    @abstractmethod
    def horde_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the environment's env_start function.
        Returns:
            The first action the horde takes.
        """

    @abstractmethod
    def horde_step(self, observation):
        """A step taken by the horde.
        Args:
            observation (int): the state observation from the
                environment's step based, where the horde ended up after the
                last step
        Returns:
            The action the horde is taking.
        """

    @abstractmethod
    def horde_end(self):
        """Runs when the horde terminates.
        """

    @abstractmethod
    def horde_cleanup(self):
        """Cleanup done after the horde ends."""

    @abstractmethod
    def horde_message(self, message):
        """A function used to pass information from the horde to the experiment.
        Args:
            message: The message passed to the horde.
        Returns:
            The response (or answer) to the message.
        """
