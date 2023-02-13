import numpy as np
import random
from keras.models import Sequential, load_model, clone_model
from keras.layers import InputLayer
from keras.layers import Dense
from tictacmap import TictactoeEnv
import tensorflow as tf

class DQNAgent():
    def __init__(self, model=None):
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

        self.env = TictactoeEnv()

        self.memory = []
        self.model = self._build_model() if model == None else model

    def _build_model(self):
        # Model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, 10)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(9, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    def load(self, name):
        model = load_model(name)

    def act(self, state, epsilon_greedy=True):
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(0, 9)
        else:
            action = np.argmax(self.model.predict(state)[0])
        return action

    def train(self, opp_fun):
        # Select starting board and number of cells filled
        init_eps = np.random.random()
        # 20 % chance of having 0 to 3 cells filled: [0 0.8) -> [0 3]
        # 5 % chance of having 4 to 7 cells filled: [0.8 1) -> [4 7]
        init_mode = int(init_eps * 5) if init_eps < 0.8 else int(init_eps * 20 - 12)
        state, _, done, _ = self.env.random_board(init_mode)

        while not done:
            # Player move
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.append((state, action, reward, next_state, done))
            if not done:
                # Opp move
                action = opp_fun()
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))

    def val(self, opp_fun):
        rewards = [0.] * 10
        for i in range(10):
            state = self.env.reset()
            done = False
            while not done:
                # Player move
                action = self.act(state, epsilon_greedy=False)
                state, reward, done, _ = self.env.step(action)
                if not done:
                    # Opp move
                    state, reward, done, _ = self.env.step(opp_fun())
                    # Player reward is negative from opponent side
                    reward = -reward
            rewards[i] = reward
        return np.mean(rewards)

    def replay(self):
        for state, action, reward, next_state, done in self.memory:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory = []

agent = DQNAgent()
val_history = [(0, agent.val(agent.env.better_valid_move))]
skipped = 0
for i in range(1, 1001):
    # Compute training
    agent.train(agent.env.valid_random_move)

    # Update weights
    if len(agent.memory) > agent.batch_size:
        agent.replay()
        val_reward = agent.val(agent.env.better_valid_move)
        best_epoch, best_reward = val_history[-1]

        print(f'Epoch {i}/{1000}. Epsilon: {agent.epsilon:.3f}. Reward: {val_reward}')
        if val_reward > best_reward:
            val_history.append((i, val_reward))
            print(f'Reward improved from {best_reward} (epoch {best_epoch})')
            agent.model.save('model_DQN_random_' + str(i))
        elif skipped >= 10:
            print(f'No improvement since {best_epoch} (reward {best_reward}), skipping to fine-tuning')
            # Early stopping
            break
        else:
            skipped += 1
            print(f'Reward not improved since epoch {best_epoch} (reward {best_reward})')

# Fine-tuning
agent.epsilon = agent.epsilon_min
opp_model = clone_model(agent.model)
for j in range(i, i + 1000):
    # Compute training
    agent.train(agent.env.AI_move(opp_model))

    # Update weights
    if len(agent.memory) > agent.batch_size:
        agent.replay()
        val_reward = agent.val(agent.env.better_valid_move)
        best_epoch, best_reward = val_history[-1]

        print(f'Epoch {j}/{1000}. Epsilon: {agent.epsilon:.3f}. Reward: {val_reward}')
        if val_reward > best_reward:
            val_history.append(i, val_reward)
            print(f'Reward improved from {best_reward} (epoch {best_epoch})')
            agent.model.save('model_DQN_vs_' + str(j))
        elif skipped >= 10:
            print(f'No improvement since {best_epoch} (reward {best_reward}), end of learning process')
            # Early stopping
            break
        else:
            skipped += 1
            print(f'Reward not improved since epoch {best_epoch} (reward {best_reward})')