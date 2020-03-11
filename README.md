# BetterRL
One day, I had a project for my Advanced RL class, and I wanted to implement two simple algorithms and compare them in a setting that looked interesting to me. There was a small catch, all implementations must be raw python, and no deep learning libraries such as tensorflow or pytorch, also no neural network for state representation was allowed. Which is amazing, because when I code from scratch usually I understand everything much better. Since I implemented the algorithms I decided to use before, I was confident that there won't be any issue, boy was I wrong... First of all, there was little to no help from internet, no one cares about linearity, deep learning is such a focus that almost no one actually uses anything else. Still it wouldn't be any issue, since my professor is Rich Sutton and this is University of Alberta, almost everyone is an RL guy, whoever you talk, would have really good knowledge in RL, so it is easy to ask someone and get answers if you stuck. I asked the issue I had, which was about the dimensions of the weight vector, and realized that everyone have their own answers. Some were true, but only hard to interpret and put it in code, and some were just poor explanations seems to be shaky even for the one who explains it. So I decided to write the RL algorithms (Starting from Chapter 10, then adding the easier ones) from scratch without neural networks, adding small tutorials, and possible mistakes or confusions that one might face with.

Anyways it might be an introductory level repository, but I believe everyone from every level of knowledge can find something for themselves, feel free to check it out, and point out my mistakes and contribute if interested.

**PS:** Don't get distracted by the number of lines for the codes, I have a lot of comments, if you start reading you will see everything is really straight forward to understand.

### Code Structure

I used a simple structure that pretty much everyone who used gym is familiar with. Also RL Glue is using a similar structure I believe. We have three components for any RL experiment;

- Environment
- Agent
- Experiment

Therefor env and agent needs to be seperated, and combined in the experiment part. I have an environment structure which is same for all the environments I implemented. There are two main functions;

- ` step(action)` : Takes an action an returns the observations, reward and if terminal state is reached or not
- ` reset()`: Returns the initial state, no action is needed.

And agent follows only three functions outside the class;

- `step(obs)`: Takes the observations and returns an action.
- ` update(observations, actions, rewards)`: Updates the agent wheather weights or tables whatever is needed depending on the algorithm.
- `end(observations, actions, rewards)`: Same as update, but only for terminal state.

So simple example would be like

```python
num_of_episodes = 100
for e in range(num_of_episodes):
  obs = env.reset()
  done = False
  while not done:
    action = agent.step(obs)
    obs, reward, done = env.step(action)
    
    observations, rewards, actions = record_data(obs, reward, action)
    
    if done:
        agent.end(observations, actions, rewards)
        break
    agent.update(observations, actions, rewards)
```

### Documentation Structure

The main source of this repository is the book, Introduction to Reinforcement Learning (Sutton and Barto, 2018). Therefor there will be two parts for the Algorithms; the first one is for the algorithms in the book, and the second one is the extra ones from the papers I found interesting.

First part also divided into two parts, Value Based and Policy Based algorithms. Naturally following from the book value-based methods also have prediction and control methods seperated. Each algorithm will mention the corresponding chapter in the book, and pages if interested in more. Otherwise I will have a tutorial linked to my blog that I try to explain from scratch as well as the code for you to re-use or read (with comments).  

Even though the main purpose of the repository is educational, it still can be used as a library that uses RL algorithm (in a linear fashion), meaning that every algorithm is implemented as efficiently as possible. To add neural networks you can modify the corresponding algorithm, should be easy at least for the *function approximation* algorithms

#### Example

I also have [\_experiment-n-step.py](https://github.com/BedirT/BetterRL/blob/master/_experiment_n-step.py) to have a general idea of how to use the library if needed. The file contains a full size experiment setting, and totally usable as well.

### Environments

| Environment Name        | Code                                          |Explanation|
| ----------------------- | --------------------------------------------- | -------- |
| **Grid World**          | [Implementation](/environments/grid_world.py) | Tutorial |
| **Cart Pole**           |                                               |          |
| **Mountain Car**        |                                               |          |
| **Rock-Paper-Scissors** |                                               |          |
| **BlackJack**           |                                               |          |

### Value-Based Algorithms

| Prediction Algorithms   | Code                                          |Explanation|
| ----------------------- | --------------------------------------------- | --------- |
| **TD(0)** 		  |						  |	      | 

| Control Algorithms      | Code                                          |Explanation|
| ----------------------- | --------------------------------------------- | -------- |
| **Episodic Semi-Gradient n-step SARSA**|[Implementation](value_based/Semi_Gradient_SARSA.py)|[Tutorial](https://bedirt.github.io/reinforcement%20learning/control%20methods/2020/03/10/Semi-Gradient-Control.html)|
| **Differential Semi-Gradient n-step SARSA**|[Implementation](value_based/Differential_Semi_Gradient_SARSA.py)|          |

### Policy-Based Algorithms

| Prediction Algorithms   | Code                                          |Explanation|
| ----------------------- | --------------------------------------------- | --------- |
| **REINFORCE**|[Implementation](policy_based/REINFORCE.py)|	      | 
| **REINFORCE with Baseline**|						  |	      | 
| **Actor-Critic** 	  |						  |	      | 

