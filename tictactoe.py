import numpy as np

class Grid:
    def __init__(self, grid_list = None):
        self.grid = np.zeros(((9,)), dtype=np.uint8)
        self.active_player = 'x'
        if grid_list is not None:
            self.grid[grid_list[9:18]] = ord('o')
            self.grid[grid_list[:9]] = ord('x')
            self.active_player = 'x' if grid_list[18] else 'o'

    def set_cell(self, cell):
        is_valid = True

        if cell not in tuple(range(0, 9)) or self.grid[cell]:
            is_valid = False
        else:
            self.grid[cell] = ord(self.active_player)
            self.active_player = 'x' if self.active_player == 'o' else 'o'

        return is_valid

    def get_cell(self, cell):
        return 'x' if self.grid[cell] else 'o' if self.grid[cell] else None

    def print(self):
        print('Active player:', self.active_player)
        print('-------')
        for i in range(3):
            print('|', end='')
            for j in range(3):
                print(chr(self.grid[3 * i + j]), end='|') if self.grid[3 * i + j] else print(' ', end='|')
            print('\n-------')

    """9 cells indicate if there is a cross in the corresponding square
    9 cells indicate if there is a circle in the corresponding square
    1 cell indicates the active player (0=cross, 1=circle)
    Repartition of squares:
    0 1 2
    3 4 5
    6 7 8"""
    def to_list(self):
        grid_list = np.zeros((19,), dtype=np.bool8)
        grid_list[:9] = self.grid == ord('x')
        grid_list[9:18] = self.grid == ord('o')
        grid_list[18] = self.active_player == 'o'

        return grid_list

    """"
    Returns the winner, None if the game is not finished.
    To check draws, use self.grid.all()
    """
    def winner(self):
        winner = None
        for i in range(3):
            if self.grid[3 * i] and self.grid[3 * i] == self.grid[3 * i + 1] == self.grid[3 * i + 2]:
                winner = chr(self.grid[3 * i])
            if self.grid[i] and self.grid[i] == self.grid[3 + i] == self.grid[6 + i]:
                winner = chr(self.grid[i])
        if self.grid[4]:
             if self.grid[0] == self.grid[4] == self.grid[8] or self.grid[2] == self.grid[4] == self.grid[6]:
                winner = chr(self.grid[0])
        return winner

    def simulation(self):
        winner = None
        while not self.grid.all():
            while not self.set_cell(np.random.randint(9)):
                pass
            winner = self.winner()
            if winner:
                print('Winner:', winner)
                self.print()
                break
        if not winner:
            print('Draw')
            self.print()

Grid().simulation()