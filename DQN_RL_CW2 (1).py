import numpy as np
import gym
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class experience: # Replay Memory
    def __init__(self, env, batch_size):
        self.env=env
        self.batch_size=batch_size
    def replay(self, replay_experience):
        ExpBatchSize = min(len(replay_experience), self.batch_size)
        SmallBatch = random.sample(replay_experience, ExpBatchSize)
        ExpStates = np.ndarray(shape = (ExpBatchSize, env.observation_space.shape[0])) 
        ExpActions = np.ndarray(shape = (ExpBatchSize, 1))
        ExpRewards = np.ndarray(shape = (ExpBatchSize, 1))
        ExpNextStates = np.ndarray(shape = (ExpBatchSize, env.observation_space.shape[0]))
        ExpDones = np.ndarray(shape = (ExpBatchSize, 1))
        idx=0
        for Sample in SmallBatch:
            ExpStates[idx] = Sample[0]
            ExpActions[idx] = Sample[1]
            ExpRewards[idx] = Sample[2]
            ExpNextStates[idx] = Sample[3]
            ExpDones[idx] = Sample[4]
            idx += 1
        return ExpStates, ExpActions , ExpRewards, ExpNextStates, ExpDones, ExpBatchSize
        
class DQNAgent:
    def __init__(self, env, optimizer, batch_size,DiscountFactor, Epsilon):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.DiscountFactor = DiscountFactor
        self.Epsilon = Epsilon
        self.replay_experience = deque(maxlen=1000000)
        self.trainstep = 0
        self.replace = 100
        self.env=env
        self.exp=experience(self.env,128)
        
        #PolicyNetwork
        self.PolicyNetwork = Sequential()
        self.PolicyNetwork.add(Dense(128, input_dim = self.env.observation_space.shape[0], activation = "relu"))
        self.PolicyNetwork.add(Dense(128 , activation = "relu"))
        self.PolicyNetwork.add(Dense(self.env.action_space.n, activation = "linear"))
        self.PolicyNetwork.compile(loss = "mse", optimizer = self.optimizer)

        #Target Network
        self.TargetNetwork = Sequential()
        self.TargetNetwork.add(Dense(128, input_dim = self.env.observation_space.shape[0], activation = "relu"))
        self.TargetNetwork.add(Dense(128 , activation = "relu"))
        self.TargetNetwork.add(Dense(self.env.action_space.n, activation = "linear"))
        self.TargetNetwork.compile(loss = "mse", optimizer = self.optimizer)

        self.TargetNetwork.set_weights(self.PolicyNetwork.get_weights())
    
    def EpsilonGreedyPolicy(self, state): # Exploration or Exploitation
        if np.random.uniform(0.0, 1.0) < self.Epsilon: 
            action = np.random.choice(self.env.action_space.n)
        else:
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            action = self.PolicyNetwork.predict(state) 
            action = np.argmax(action[0]) 
            
        return action
    
    def SaveModel(self):
        self.PolicyNetwork.save('DQN_PolicyModel.h5')
        self.TargetNetwork.save('DQN_TargetModel.h5')

    def LoadModel(self):
        self.PolicyNetwork.set_weights(tf.keras.models.load_model('DQN_PolicyModel.h5').get_weights())
        self.TargetNetwork.set_weights(tf.keras.models.load_model('DQN_TargetModel.h5').get_weights())
     
    # Train the Optimal and Target Neural Network with the sample from replay memory
    def train(self):
        ExpStates, ExpActions , ExpRewards, ExpNextStates, ExpDones,ExpBatchSize= self.exp.replay(self.replay_experience)
        QNext = self.TargetNetwork.predict(ExpNextStates)
        QNext = QNext * (np.ones(shape = ExpDones.shape) - ExpDones)
        QNext = np.max(QNext, axis=1)
        QCurrent = self.PolicyNetwork.predict(ExpStates)
        for i in range(ExpBatchSize):
            a = ExpActions[i,0]
            QCurrent[i,int(a)] = ExpRewards[i] + self.DiscountFactor * QNext[i] 
        QTarget = QCurrent
        self.PolicyNetwork.fit(ExpStates, QTarget, epochs = 1, verbose = 0)

#Intialisation of the variables
env=gym.make("LunarLander-v2")
RLagent = DQNAgent(env, optimizer=Adam(learning_rate = 0.0001), batch_size = 128, DiscountFactor= 0.99, Epsilon=1.0)
#RLagent.LoadModel()
rewards = []
average_reward = []
average = deque(maxlen=100)
EpisodeSize=1000

# Training the RL agent for x amount of episodes
for episode in range(EpisodeSize):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = RLagent.EpsilonGreedyPolicy(state)
        next_state, reward, done, info = env.step(action)
       # env.render()
        total_reward += reward
        RLagent.replay_experience.append((state, action, reward, next_state, done)) # add the current experience to the replay memory
        RLagent.train() #Train the neural networks with the experience in the replay memory 
        state = next_state 
    average.append(total_reward)     
    average_reward.append(np.mean(average))
    rewards.append(total_reward)
    
    # update model_target after each episode
   # if episode % 5==0:
        #print("Training step is ",episode)
    RLagent.TargetNetwork.set_weights(RLagent.PolicyNetwork.get_weights())
    RLagent.Epsilon = max(0.1, 0.995 * RLagent.Epsilon) 
    print("Episode={} Reward={}".format(episode, total_reward))

#Plot the learning curve for the reward and average reward
plt.title("Agent Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(rewards,label ='rewards' )
plt.plot(average_reward,label ='average rewards' )
plt.legend()
plt.show()

#Save the weights after the training is done
RLagent.SaveModel()

    