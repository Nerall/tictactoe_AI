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
        self.epsilon_decay = 0.99
        self.batch_size = 32

        self.env = TictactoeEnv()

        self.memory = []
        self.model = self._build_model() if model == None else model

    def _build_model(self):
        # Model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, 9)))
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

    def train(self, opp_fun, args=None):
        # Select starting board and number of cells filled
        init_eps = np.random.random()
        # 20 % chance of having 0 to 3 cells filled: [0 0.8) -> [0 3]
        # 5 % chance of having 4 to 7 cells filled: [0.8 1) -> [4 7]
        # init_mode = int(init_eps * 5) if init_eps < 0.8 else int(init_eps * 20 - 12)
        # TODO
        init_mode = int(init_eps < 0.5)
        state, _, done, _ = self.env.random_board(init_mode)

        while not done:
            # Player move
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
            if not done:
                # Opp move
                if not args:
                    action = opp_fun()
                else:
                    action = opp_fun(args)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

    def val(self, opp_fun):
        rewards = [0.] * 50
        for i in range(50):
            state, reward, done = self.env.reset(), 0, False
            if i % 2:
                # Switch between 'X' and 'O' player
                state, reward, done, _ = self.env.step(opp_fun())
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
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
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
val_history = [(0, -100.)]
for i in range(1, 1001):
    # Compute training
    while len(agent.memory) < 1000:
        agent.train(agent.env.valid_random_move)

    # Update weights
    agent.replay()

    val_reward = agent.val(agent.env.better_valid_move)
    best_epoch, best_reward = val_history[-1]

    print(f'Epoch {i}/{1000}. Epsilon: {agent.epsilon:.3f}. Reward: {val_reward:.3f}')
    if val_reward > best_reward:
        val_history.append((i, val_reward))
        print(f'  Reward improved from {best_reward:.3f} (epoch {best_epoch})')
        agent.model.save('model_DQN_random_' + str(i))
    elif i >= best_epoch + 50:
        print(f'  No improvement since {best_epoch} (reward {best_reward:.3f}), skipping to fine-tuning')
        # Early stopping
        break
    else:
        print(f'  Reward not improved since epoch {best_epoch} (reward {best_reward:.3f})')

# Fine-tuning
best_model_name = 'model_DQN_random_' + str(best_epoch)
print('Loading back model', best_model_name)
agent = DQNAgent(load_model(best_model_name))
agent.epsilon = agent.epsilon_min
opp_model = load_model(best_model_name)
val_history = [(0, -100.)]
for i in range(1, 1001):
    # Compute training
    while len(agent.memory) < 1000:
        agent.train(agent.env.AI_move, opp_model)

    # Update weights
    agent.replay()
    val_reward = agent.val(agent.env.better_valid_move)
    best_epoch, best_reward = val_history[-1]

    print(f'Epoch {i}/{1000}. Epsilon: {agent.epsilon:.3f}. Reward: {val_reward:.3f}')
    if val_reward > best_reward:
        val_history.append((i, val_reward))
        print(f'Reward improved from {best_reward:.3f} (epoch {best_epoch})')
        agent.model.save('model_DQN_vs_' + str(i))
    elif i >= best_epoch + 50:
        print(f'No improvement since {best_epoch} (reward {best_reward:.3f}), end of learning process')
        # Early stopping
        break
    else:
        print(f'Reward not improved since epoch {best_epoch} (reward {best_reward:.3f})')