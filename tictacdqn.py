import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from tictacmap import TictactoeEnv

env = TictactoeEnv()

discount_factor = 0.99
eps = 0.5
eps_decay_factor = 0.999
num_episodes=2001

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    if i and i % 10 == 0:
        print(i, eps, sum(rewards[i - 10: i]))
    if i and i % 250 == 0:
        print(i, eps, sum(rewards[i - 250: i]))
        model.save('model_' + str(i))
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(model.predict(state.reshape(1, -1), verbose=0))

        new_state, reward, done, _ = env.step(action)

        if not done:
            # Opponent move
            new_state, reward, done, _ = env.random_move()
            # Opponent reward
            reward = -reward

        target = reward + discount_factor * np.max(model.predict(new_state.reshape(1, -1), verbose=0))
        target_vector = model.predict(state.reshape(1, -1), verbose=0)[0]
        target_vector[action] = target
        model.fit(state.reshape(1, -1), 
          target_vector.reshape(-1, env.action_space.n), 
          epochs=1, verbose=0)
        state = new_state
        rewards[i] = reward
