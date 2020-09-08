import numpy as np
import tensorflow as tf


class SeparateModel:
    def __init__(self, state_dim, num_actions=4, num_quantiles=51, kappa=1, eta=1e-3, epsilon_adam = 1e-7):
        """ SeparateModel Class

        Args:
            state_dim (int) : the dimension of the state space
            num_actions (int) : the number of actions
            num_quantiles (int): the number of quantiles
            kappa (float): the parameter of the quantile huber loss
            eta (float): the learning rate
            epsilon_adam (float): the epsilon parameter of the Adam optimizer

        Initialises :
            self.tau (tf array): the quantile levels
            self.opt (keras optimizer): Adam Optimizer
            self.model (tf Sequentials): A Neural Network model
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        self.eta = eta
        self.tau = tf.constant((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), dtype=tf.float32)
        self.epsilon_adam = epsilon_adam
        self.opt = tf.keras.optimizers.Adam(self.eta, epsilon= self.epsilon_adam)
        self.model = self.create_model()
        # tensorboard
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    def create_model(self):
        """
        Returns :
            model : A Dense Neural Network model with first hidden layers of shape (128, 64, 32).
        """
        return tf.keras.models.Sequential([
            #64, 16
            tf.keras.layers.Dense(128, activation='relu', input_shape=[self.state_dim, ]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.num_actions * self.num_quantiles),
            tf.keras.layers.Reshape((self.num_actions, self.num_quantiles))])

    @tf.function
    def quantile_huber_loss(self, target, pred, actions):
        """
        We note N the number of quantiles, A the number of actions and B the batch_size.
        Args:
            target (Numpy array): the theta values of the target with shape (B, N)
            pred (Numpy array): the theta values of (last_s) with shape (B, A, N)
            actions (Numpy array) : the last_actions with shape (B, A)
        Returns:
            Computes the quantile huber loss for these samples
        """
        # pred: (B, A, N) -> (B, N)
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        # target_tile: (B, N) -> (N, B, 1)
        target_tile = tf.expand_dims(tf.transpose(target), axis=-1)
        target_tile = tf.cast(target_tile, tf.float32)
        # u: (N, B, N)
        u = target_tile - pred
        # huber_loss: (N, B, N)
        huber_loss = tf.where(tf.math.abs(u) < self.kappa,
                              1 / 2 * tf.math.pow(u, 2),
                              self.kappa * (tf.math.abs(u) - 1 / 2 * self.kappa))
        loss = huber_loss * tf.math.abs(self.tau- tf.where(u < 0, 1.0, 0.0))
        # loss -> (B, N, N)
        loss = tf.transpose(loss, [1, 0, 2])
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=1), axis=-1))
        # loss : (1)
        return loss

    #def quantile_huber_loss(self, target, pred, actions):
    #    pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
    #    pred_tile = tf.expand_dims(pred, axis=1)

    #    target_tile = tf.tile(tf.expand_dims(
    #        target, axis=2), [1, 1, self.num_quantiles])
    #    target_tile = tf.cast(target_tile, tf.float32)

    #    u = target_tile - pred_tile
    #    huber_loss = tf.where(tf.less(tf.math.abs(u), self.kappa),
    #                          1 / 2 * tf.math.pow(u, 2),
    #                          self.kappa * (tf.math.abs(u) - 1 / 2 * self.kappa))

    #    tau = tf.reshape(np.array(self.tau), [1, 1, self.num_quantiles])

    #    loss = tf.math.abs(tau - tf.where(u < 0, 1.0, 0.0))*huber_loss
    #    loss = tf.reduce_mean(tf.reduce_sum(
    #        tf.reduce_mean(loss, axis=-1), axis=-1))
    #    return loss

    @tf.function
    def average_loss(self, targets, pred, actions, pi):
        """
        We note N the number of quantiles, A the number of actions and B the batch_size.
        Args:
            targets (Numpy array): the theta values of the targets with shape (B, A, N)
            pred (Numpy array): the theta values of (last_s) with shape (B, A, N)
            actions (Numpy array) : the last_actions with shape (B, A)
            pi (Numpy array): the policy evaluated as (current_s) with shape (B, A)
        Returns:
            Computes the quantile huber loss averaged over actions for these samples
                """
        # pred: (B, A, N) -> (B, N)
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        # targets_tile: (B, A, N) -> (N, A, B, 1)
        targets_tile = tf.expand_dims(tf.transpose(targets), axis=-1)
        targets_tile = tf.cast(targets_tile, tf.float32)
        # u: (N, A, B, N)
        u = targets_tile - pred
        # huber_loss: (N, A, B, N)
        huber_loss = tf.where(tf.math.abs(u) < self.kappa,
                              1 / 2 * tf.math.pow(u, 2),
                              self.kappa * (tf.math.abs(u) - 1 / 2 * self.kappa))
        loss = huber_loss * tf.math.abs(self.tau - tf.where(u < 0, 1.0, 0.0))
        # loss -> (B, A, N, N)
        loss = tf.transpose(loss, [2, 1, 0, 3])
        loss = tf.reduce_sum(
            tf.reduce_mean(loss, axis=-1), axis=-1)
        loss = tf.reduce_mean(tf.reduce_sum(pi * loss, -1))
        return loss

    @tf.function
    def train(self, states, target, actions):
        """
        We note N the number of quantiles, A the number of actions, B the batch_size and
         S the dimension of the state space.
        Computes the quantile huber loss, and takes an optimizer step.
        Args:
            states (Numpy array): the vector representation of the last states (B, S)
            target (Numpy array) : the theta values of the target with shape (B, N)
            actions (Numpy array) : the last_actions with shape (B, A)
            """
        with tf.GradientTape() as tape:
            theta = self.model(states)
            loss = self.quantile_huber_loss(target, theta, actions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def avg_loss_train(self, states, targets, actions, pi):
        """
        We note N the number of quantiles, A the number of actions, B the batch_size and
         S the dimension of the state space.
        Computes the averaged quantile huber loss, and takes an optimizer step.
        Args:
            states (Numpy array): the vector representation of the last states (B, S)
            targets (Numpy array): the theta values of the target with shape (B, A, N)
            actions (Numpy array): the last_actions with shape (B, A)
            pi (Numpy array): the policy evaluated as (current_s) with shape (B, A)
            """
        with tf.GradientTape() as tape:
            theta = self.model(states)
            loss = self.average_loss(targets, theta, actions, pi)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss(loss)

    def predict(self, state):
        """
        Args:
            state (Numpy array): vector representation of the last state
        Returns :
            theta (Tensorflow): quantiles of the last state
        """
        if len(state.shape) == 1:
            state = state.reshape(1, self.state_dim)
        return self.model.predict(state)
