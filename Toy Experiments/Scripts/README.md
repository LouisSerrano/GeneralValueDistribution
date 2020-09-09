To implement the Hordes, we proposed the RLGLueHorde framework which adapts the RLGlue framework to our needs.

The RLGLue framework is a general solution to glue an environment with an agent, so that interactions between them occur. For instance, the environment can be a maze, and the Agent can be a Q-Learning agent. Concretely, an RLGlue instance is created with an Environment class type and Agent class type, on the basis of environment and agent parameters. This RLGlue instance then creates the Environment instance and Agent instance.

In this folder, you will find:
> environment_horde.py: an abstract class for a horde environment

> horde.py: an abstract class for a horde

> rl_glue_horde.py: a class that implements the RLGlueHorde class that glues a horde and a horde environment

> toy_env_horde.py: the toy horde environment used for the toy experiments

> utils.py: some helper functions + the transition_gen functions used to generate cumulants and gammas

> separateHorde: folder that includes a horde, GVD and model files

> unifiedHorde: folder that includes a horde, GVD and model files

> oldHorde: folder that includes a horde, valueGVF, actionValueGVF files

> MC: a monteCarloHorde, a monteCarloGVD files
