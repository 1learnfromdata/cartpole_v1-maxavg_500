import gym
import random
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('cartpole_maxavg.h5')



env = gym.make("CartPole-v1")
env.reset()
goal_steps = 1000


scores = []
choices = []
for each_game in range(3):
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

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))


