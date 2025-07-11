import numpy as np
import random

# Types de cellules
EMPTY = 0
OBSTACLE = 1
FOOD = 2      # Nourriture pour restaurer la faim
TRAP = 3      # Piège mortel
PREDATOR = 4  # Prédateur
SQUIRREL = 5  # Écureuil
RUBY = 6      # Objectif final

# Emojis pour l'affichage
CELL_EMOJI = {
        EMPTY: "🟫",    # Terre
        OBSTACLE: "🪨",
        FOOD: "🍎",     # Nourriture
        TRAP: "🧨",     # Piège
        PREDATOR: "🐺", # Prédateur
        SQUIRREL: "🐿️", # Écureuil
        RUBY: "🍄"      # Ruby (objectif)
}

class RubyQuest:
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.used_positions = set()

        self.squirrel_pos = self.random_position()
        self.used_positions.add(self.squirrel_pos)

        self.predators = []
        for _ in range(2):
            pos = self.random_position()
            self.predators.append(pos)
            self.used_positions.add(pos)

        self.ruby_pos = self.random_position()
        self.used_positions.add(self.ruby_pos)
        self.grid[self.ruby_pos] = RUBY

        for _ in range(5):
            pos = self.random_position()
            self.grid[pos] = FOOD
            self.used_positions.add(pos)

        for _ in range(3):
            pos = self.random_position()
            self.grid[pos] = TRAP
            self.used_positions.add(pos)

        for _ in range(8):
            pos = self.random_position()
            self.grid[pos] = OBSTACLE
            self.used_positions.add(pos)

        self.hunger = 100
        self.max_hunger = 100
        self.has_ruby = False
        self.done = False

    def random_position(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.used_positions:
                return pos

    def is_valid_move(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.grid[x, y] != OBSTACLE:
                return True
        return False

    def move_predators(self):
        new_predators = []
        for px, py in self.predators:
            sx, sy = self.squirrel_pos
            dx = 1 if sx > px else -1 if sx < px else 0
            dy = 1 if sy > py else -1 if sy < py else 0
            nx, ny = px + dx, py + dy
            if self.is_valid_move(nx, ny) and (nx, ny) not in self.predators:
                new_predators.append((nx, ny))
            else:
                new_predators.append((px, py))
        self.predators = new_predators

    def step(self, action):
        if self.done:
            return

        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        x, y = self.squirrel_pos
        nx, ny = x + dx, y + dy

        if self.is_valid_move(nx, ny):
            self.squirrel_pos = (nx, ny)
        else:
            return

        self.hunger -= 1

        cell = self.grid[nx, ny]
        if cell == FOOD:
            self.hunger = min(self.max_hunger, self.hunger + 30)
            self.grid[nx, ny] = EMPTY
        elif cell == TRAP:
            self.done = True
        elif (nx, ny) == self.ruby_pos:
            self.has_ruby = True
            self.grid[nx, ny] = EMPTY
            self.done = True

        if self.hunger <= 0:
            self.done = True

        self.move_predators()

        if self.squirrel_pos in self.predators:
            self.done = True

    def get_grid(self):
        display = np.full((self.grid_size, self.grid_size), CELL_EMOJI[EMPTY])
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x,y] == OBSTACLE:
                    display[x,y] = CELL_EMOJI[OBSTACLE]
                elif self.grid[x,y] == FOOD:
                    display[x,y] = CELL_EMOJI[FOOD]
                elif self.grid[x,y] == TRAP:
                    display[x,y] = CELL_EMOJI[TRAP]
                elif self.grid[x,y] == RUBY:
                    display[x,y] = CELL_EMOJI[RUBY]
        for px, py in self.predators:
            display[px, py] = CELL_EMOJI[PREDATOR]
        sx, sy = self.squirrel_pos
        display[sx, sy] = CELL_EMOJI[SQUIRREL]
        return display