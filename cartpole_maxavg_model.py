import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time


env = gym.make("CartPole-v1")
env.reset()
goal_steps = 1000
score_requirement = 120
initial_games = 1000000

training_data = []
# all scores:
scores = []
# just the scores that met our threshold:
accepted_scores = []
# iterate through however many games we want:
for _ in range(initial_games):
    score = 0
    # moves specifically from this environment:
    game_memory = []
    # previous observation that we saw
    prev_observation = []
    # for each frame in 200
    for _ in range(goal_steps):
        # choose random action (0 or 1)
        action = random.randrange(0,2)
        # do it!
        observation, reward, done, info = env.step(action)

        # notice that the observation is returned FROM the action
        # so we'll store the previous observation here, pairing
        # the prev observation to the action we'll take.
        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
        prev_observation = observation
        score+=reward
        if done: break

    # IF our score is higher than our threshold, we'd like to save
    # every move we made
    # NOTE the reinforcement methodology here.
    # all we're doing is reinforcing the score, we're not trying
    # to influence the machine in any way as to HOW that score is
    # reached.
    if score >= score_requirement:
        accepted_scores.append(score)
        for data in game_memory:
            # convert to one-hot (this is the output layer for our neural network)
            if data[1] == 1:
                output = [0,1]
            elif data[1] == 0:
                output = [1,0]

            # saving our training data
            training_data.append([data[0], output])

    # reset env to play again
    env.reset()
    # save overall scores
    scores.append(score)

# just in case you wanted to reference later
training_data_save = np.array(training_data)
np.save('saved.npy',training_data_save)
	
# some stats here, to further illustrate the neural network magic!
print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))
print("zero count",zero_count,"one count",one_count)


X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
y = [i[1] for i in training_data]
y=np.array(y)


adam = Adam(learning_rate=0.001)

model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y,batch_size=256, epochs=5,validation_split=0.2)


scores = []
choices = []
for each_game in range(10000):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        #env.render()
        #time.sleep(1)

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)
    print(score)

# Save model
model.save('cartpole_maxavg.h5')
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)

