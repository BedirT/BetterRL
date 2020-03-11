import numpy as np

class REINFORCE:
    def __init__(self, feature_space, action_space, alpha = 0.0001, gamma = 0.99):
        self.alpha = alpha
        self.gamma = gamma
    
        self.feature_space = feature_space
        self.action_space = action_space

        self.reset_weights()

    def step(self, obs):
        '''
        Picks the action given the observations with the probabilities
        of each action
        '''
        # Gets the probability for each action
        probs = self._policy(obs)

        # Pick the action using probabilities
        action = np.random.choice(self.action_space, p=probs)
        return action

    def _policy(self, obs):
        '''
        Probabilities are calculated using softmax.
        So for each action
            -> e^{h(s,a,theta)}/sum_b{e^{h(s,b,theta)}}

        Returns to the probabilitiess of each action being taken
        '''
        probs = np.zeros(self.action_space)
        for a in range(self.action_space):
            probs[a] = self._h(obs, a)
        probs = np.exp(probs)
        return probs/np.sum(probs)

    def _gradient(self, obs, action):
        '''
        Softmax gradient calculated given observations and action
        '''
        grads = np.zeros_like(self.theta)
        probs = self._policy(obs)
        for b in range(self.action_space):
            # sum_b{x(s,b) * π(b|s, theta)}
            grads += self._x(obs, b) * probs[b]
        # returns x(s,a) - grads
        return self._x(obs, action) - grads

    def _h(self, obs, action):
        '''
        h(s, a, theta) from the book. Here we just have a linear
        function so its theta^T.x(s,a). 
        '''
        return self.theta.T.dot(self._x(obs, action))

    def _x(self, obs, action):
        '''
        x(s,a). State representation is created here. We have the same state representation
        for each action, except that only the selected action will have them activated meaning
        the rest will be just zeros in our case (since we are using one hot vectors for the
        representations)

        -> Return to the representation created i.e. [0 0 0 0 0 1 0 0 0 0 0 0] 
        '''
        one_hot = np.zeros_like(self.theta)
        # need to improve this
        j = 0
        for i in range(action * self.feature_space, ((action+1) * self.feature_space)):
            one_hot[i] = obs[j]
            j += 1
        return one_hot

    def update(self, observations, actions, rewards):
        '''
        Since REINFORCE is a Monte Carlo method, there will be no
        gradual updating until the Terminal state
        '''
        pass

    def end(self, observations, actions, rewards):
        '''
        Updates the weights given the observations actions and rewards.
        Formula is:
            -> theta = theta + alpha x ∆π(s,a) x G
                    where ∆ is the gradient
                    and G is the discounted future rewards
        '''
        for i in range(len(observations)):
            G = sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])
            self.theta += self.alpha * self._gradient(observations[i], actions[i]) * G
                 
    def reset_weights(self):
        '''
        Resets the weights
        Weights dimension is d x a where a is action space and d is the feature space
        - Think of it as one one hot vector for each action
        '''
        self.theta = np.random.rand(self.feature_space * self.action_space)
        
        