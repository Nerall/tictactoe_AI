import gym
from gym import spaces
import pygame
import numpy as np

class TictactoeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=3):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(9)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset()
        self.board = np.zeros((9,), dtype=np.int8)
        self.active_player = 'X'

        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs()

    def _get_obs(self):
        return self.board.reshape(1,  -1)
        # TODO
        # return np.append(self.board, self.active_player != 'X').reshape(1, -1)

    def _winner(self):
        winner = None
        for i in range(3):
            if self.board[3 * i] and self.board[3 * i] == self.board[3 * i + 1] == self.board[3 * i + 2]:
                winner = {1:'X', -1:'O'}[self.board[3 * i]]
            if self.board[i] and self.board[i] == self.board[3 + i] == self.board[6 + i]:
                winner = {1:'X', -1:'O'}[self.board[i]]
        if self.board[4]:
             if self.board[0] == self.board[4] == self.board[8] or self.board[2] == self.board[4] == self.board[6]:
                winner = {1:'X', -1:'O'}[self.board[4]]
        return winner

    def step(self, action):
        """
        Returns:
            list: observations
            int: reward
            bool: done
            dict: additional information
        """
        # Invalid move
        if not self.action_space.contains(action) or self.board[action]:
            return self._get_obs(), -1., True, None

        reward = 0.
        done = False

        self.board[action] = {'X':1, 'O':-1}[self.active_player]
        winner = self._winner()
        if winner:
            reward = 1. - 0.01 * self.board[self.board != 0].size
            if winner != self.active_player:
                reward = -reward
            done = True
        elif self.board.all(): # Draw
            done = True
        self.active_player = {'X':'O', 'O':'X'}[self.active_player]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, None

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size
        if self.board.any():
            for i, cell in enumerate(self.board):
                size = pix_square_size * 0.4
                # Left to right, top to bottom
                posX = pix_square_size * (0.5 + i % 3)
                posY = pix_square_size * (0.5 + i // 3)
                if cell == 1:
                    pygame.draw.line(canvas, "blue", (posX - size, posY - size), (posX + size, posY + size), 5)
                    pygame.draw.line(canvas, "blue", (posX + size, posY - size), (posX - size, posY + size), 5)
                elif cell == 2:
                    pygame.draw.circle(canvas, "red", (posX, posY), size, 5)

        for i in range(self.size + 1):
            pygame.draw.line(canvas, "black", (0, i * pix_square_size), (self.window_size, i * pix_square_size), 3)
            pygame.draw.line(canvas, "black", (i * pix_square_size, 0), (i * pix_square_size, self.window_size), 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def random_move(self):
        action = np.random.randint(0, 9)
        return action

    def valid_random_move(self):
        valid_cells = np.arange(9)[self.board == 0]
        action = np.random.choice(valid_cells)
        return action

    def better_valid_move(self):
        player = {'X':1, 'O':-1}[self.active_player]
        opp = 'X' if player == 'O' else 'O'
        action = None

        lines = [(i, i + 1, i + 2) for i in range(3)] \
              + [(i, i + 3, i + 6) for i in range(3)] \
              + [(0, 4, 8), (2, 4, 6)]
        # Instant win
        for line in lines:
            player_cells = [idx for idx in line if self.board[idx] == player]
            if len(player_cells) == 2:
                action = list(set(line) - set(player_cells))[0]
                break
        if action == None:
            # Instant lose
            for line in lines:
                opp_cells = [idx for idx in line if self.board[idx] == opp]
                if len(opp_cells) == 2:
                    action = list(set(line) - set(opp_cells))[0]
                    break
            if action == None:
                # Random move
                action = self.valid_random_move()

        return action

    def human_move(self):
        player_input = input()
        assert player_input.isdigit()
        action = int(player_input)
        return action

    def AI_move(self, model):
        action = np.argmax(model.predict(self._get_obs())[0])
        return action

    def random_board(self, played_cells):
        if not (0 <= played_cells <= 8):
            return None

        self.reset()
        obs, reward, done, info = self._get_obs(), 0, False, None
        for i in range(min(5, played_cells)):
            # No lock if 5 moves or less
            if i < 5:
                tmp_board = np.copy(self.board)
                tmp_player = self.active_player
            action = self.valid_random_move()
            obs, reward, done, info = self.step(action)
            while done:
                # Revert action
                self.board = np.copy(tmp_board)
                self.active_player = tmp_player
                action = self.valid_random_move()
                obs, reward, done, info = self.step(action)

        return obs, reward, done, info