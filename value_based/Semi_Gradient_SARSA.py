import numpy as np
import random

class SG_SARSA:
    def __init__(self, feature_space, action_space, n=8, alpha = 0.0001, gamma = 0.99, eps = .1):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.n = n
    
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
            q_vals[a] = self._q_hat(obs, a)
        return q_vals

    def _q_hat(self, obs, action):
        '''
        q(s, a). Calculates the q values of the action
        given the observations.
        -> returns q(s, a) = w^T.x(s,a)
                since we are using linear function approximations
        '''
        return self.w.T.dot(self._x(obs, action))

    def _grad_q_hat(self, obs, action):
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
        Updating the weights. Since this is n-step update, what we do is simply
        keeping only n+1 elements in the trajectory and removing the elements
        from beginning since we won't be using them anymore.
        '''
        
        # Checking if we have more than n+1 elements, if so we will remove the first element
        # Only possible size is n+2, so we would be making it n+1 again.
        if len(observations) > self.n+1:
            observations.pop(0)
            rewards.pop(0)
            actions.pop(0)

        # First checking if there are enough elements to make the update
        if len(rewards) == self.n+1:
            # w_{t+1} = w_t + alpha x [G - q_hat(s_0, a_0)] grad_q(s_0, a_0)
            #               where G = sum_{t=0}^{n-1}(gamma^t) * R_t + [gamma^n * q_hat(s_{n}, a_{n})]
            G = sum([(self.gamma ** t) * r for t,r in enumerate(rewards[:-1])])
            G += (self.gamma ** (self.n)) * self._q_hat(observations[-1], actions[-1])
            self.w += self.alpha * (G - self._q_hat(observations[0], actions[0])) * \
                self._grad_q_hat(observations[0], actions[0])

    def end(self, observations, actions, rewards):
        '''
        Should be called when the terminal state is reached, it is the update function
        for the terminal state.

        For this algorithm, we are removing elements while updating w according to the discounted
        rewards only.
        '''
        for _ in range(self.n):
            observations.pop(0)
            rewards.pop(0)
            actions.pop(0)

            G = sum([(self.gamma ** t) * r for t,r in enumerate(rewards)])
            self.w += self.alpha * (G - self._q_hat(observations[0], actions[0])) * self._grad_q_hat(observations[0], actions[0])

    def reset_weights(self):
        '''
        Resets the weights
        Weights dimension is d x a where a is action space and d is the feature space
        - Think of it as one one hot vector for each action
        '''
        self.w = np.zeros((self.feature_space * self.action_space))
