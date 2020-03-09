import numpy as np
import random

class SG_SARSA_MonteCarlo:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99, eps = .1):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
        self.feature_space = feature_space
        self.action_space = action_space

        self.reset_weights()

    def step(self, obs):
        '''
        Picks the action given the observations using epsilon-greedy
        action selection.
        '''
        if np.random.sample() > self.eps:
            return np.argmax(self._act(obs))
        else:
            return np.random.randint(0, self.action_space)

    def _act(self, obs):
        '''
        Gets the q-values for each action given the observations.
        '''
        q_vals = np.zeros(self.action_space)
        for a in range(self.action_space):
            q_vals[a] = self.q_hat(obs, a)
        return q_vals

    def q_hat(self, obs, action):
        '''
        q(s, a). Calculates the q values of the action
        given the observations.
        -> returns q(s, a) = w^T.x(s,a)
                since we are using linear function approximations
        '''
        return self.w.T.dot(self._x(obs, action))

    def grad_q_hat(self, obs, action):
        '''
        Gradient of q(s,a)
        In out case since we are using linear functions;
        Derivative of w^T.x(s,a) -> x(s,a)
        '''
        return self._x(obs, action)

    def _x(self, obs, action):
        '''
        x(s,a). State representation is created here. We have the same state representation
        for each action, except that only the selected action will have them activated meaning
        the rest will be just zeros in our case (since we are using one hot vectors for the
        representations)

        -> Return to the representation created i.e. [0 0 0 0 0 1 0 0 0 0 0 0] 
        '''
        one_hot = np.zeros_like(self.w)
        j = 0
        for i in range(action * self.feature_space, ((action+1) * self.feature_space)):
            one_hot[i] = obs[j]
            j += 1
        return one_hot

    def update(self, observations, actions, rewards):
        '''
        Updates the weights given the observations actions and rewards.
        Formula is:
            -> w = w + alpha x (G - q(s,a)) x ∆q(s,a)
                    where ∆ is the gradient
                    and G is the discounted future rewards
        '''
        for i in range(len(observations)):
            G = sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])
            self.w += self.alpha * self.grad_q_hat(observations[i], actions[i]) * (G - self.q_hat(observations[i], actions[i]))

    def reset_weights(self):
        '''
        Resets the weights
        Weights dimension is d x a where a is action space and d is the feature space
        - Think of it as one one hot vector for each action
        '''
        self.w = np.random.rand(self.feature_space * self.action_space)