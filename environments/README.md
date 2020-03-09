# BetterRL
One day, I had a project for my Advanced RL class, and I wanted to implement two simple algorithms and compare them in a setting that looked interesting to me. There was a small catch, all implementations must be raw python, and no deep learning libraries such as tensorflow or pytorch, also no neural network for state representation was allowed. Which is amazing, because when I code from scratch usually I understand everything much better. Since I implemented the algorithms I decided to use before, I was confident that there won't be any issue, boy was I wrong... First of all, there was little to no help from internet, no one cares about linearity, deep learning is such a focus that almost no one actually uses anything else. Still it wouldn't be any issue, since my professor is Rich Sutton and this is University of Alberta, almost everyone is an RL guy, whoever you talk, would have really good knowledge in RL, so it is easy to ask someone and get answers if you stuck. I asked the issue I had, which was about the dimensions of the weight vector, and realized that everyone have their own answers. Some were true, but only hard to interpret and put it in code, and some were just poor explanations seems to be shaky even for the one who explains it. So I decided to write the RL algorithms (Starting from Chapter 10, then adding the easier ones) from scratch without neural networks, adding small tutorials, and possible mistakes or confusions that one might face with.

Anyways it might be an introductory level repository, but I believe everyone from every level of knowledge can find something for themselves, feel free to check it out, and point out my mistakes and contribute if interested.

## Environments

- [x] Grid World
- [ ] Cart Pole
- [ ] Mountain Car
- [ ] Rock-Paper-Scissors

## Value-Based Algorithms

#### Prediction

#### Control

- [x] Episodic Semi-Gradient SARSA
- [ ] Episodic Semi-Gradient n-step SARSA
- [ ] Differential Semi-Gradient SARSA
- [ ] Differential Semi-Gradient n-step SARSA

## Policy Gradient Algorithms

#### Control

- [x] REINFORCE
- [ ] REINFORCE with Baseline
- [ ] Actor-Critic

