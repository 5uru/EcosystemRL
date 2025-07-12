import numpy as np
import random

# Types de cellules
EMPTY = 0
OBSTACLE = 1
FOOD = 2      # Nourriture
TRAP = 3      # Piège
PREDATOR = 4  # Prédateur
SQUIRREL = 5  # Écureuil
RUBY = 6      # Ruby

# Paramètres RL
MAX_HUNGER = 100
GRID_SIZE = 10

class RubyQuestRL:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        """Réinitialise l'environnement"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.used_positions = set()

        # Paramètres
        self.hunger = MAX_HUNGER
        self.has_ruby = False
        self.done = False

        # Générer l'écureuil
        self.squirrel_pos = self.random_position()
        self.used_positions.add(self.squirrel_pos)

        # Générer le ruby
        self.ruby_pos = self.random_position()
        self.used_positions.add(self.ruby_pos)
        self.grid[self.ruby_pos] = RUBY

        # Générer les prédateurs (hors zone critique)
        self.predators = []
        for _ in range(2):  # 2 prédateurs
            pos = self.random_position()
            while self.is_too_close(pos):
                pos = self.random_position()
            self.predators.append(pos)
            self.used_positions.add(pos)

        # Générer la nourriture
        self.foods = []
        for _ in range(5):
            pos = self.random_position()
            self.foods.append(pos)
            self.used_positions.add(pos)
            self.grid[pos] = FOOD

        # Générer les pièges
        self.traps = []
        for _ in range(3):
            pos = self.random_position()
            self.traps.append(pos)
            self.used_positions.add(pos)
            self.grid[pos] = TRAP

        # Générer les obstacles
        self.obstacles = []
        for _ in range(8):
            pos = self.random_position()
            self.obstacles.append(pos)
            self.used_positions.add(pos)
            self.grid[pos] = OBSTACLE

        return self.get_state()

    def random_position(self):
        while True:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in self.used_positions:
                return pos

    def is_too_close(self, pos):
        """Évite les prédateurs trop proches de l'écureuil ou du ruby"""
        sx, sy = self.squirrel_pos
        rx, ry = self.ruby_pos
        distance_squirrel = abs(pos[0]-sx) + abs(pos[1]-sy)
        distance_ruby = abs(pos[0]-rx) + abs(pos[1]-ry)
        return distance_squirrel <= 3 or distance_ruby <= 3

    def is_valid_move(self, x, y):
        return (0 <= x < self.grid_size and
                0 <= y < self.grid_size and
                (x, y) not in self.obstacles)

    def move_predators(self):
        """Mouvement des prédateurs (vers écureuil si proche)"""
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
        """Exécute une action et retourne (état, récompense, done)"""
        if self.done:
            return self.get_state(), 0, True

        # Mouvement (0=Haut, 1=Droite, 2=Bas, 3=Gauche)
        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        x, y = self.squirrel_pos
        nx, ny = x + dx, y + dy

        # Vérifier si mouvement valide
        if not self.is_valid_move(nx, ny):
            return self.get_state(), -2, False

        self.squirrel_pos = (nx, ny)
        self.hunger -= 1

        # Récompense de base
        reward = -0.01  # Pénalité par pas

        # Interaction avec la case
        if self.squirrel_pos in self.foods:
            self.hunger = min(MAX_HUNGER, self.hunger + 30)
            self.foods.remove(self.squirrel_pos)
            reward += 1.0  # Récompense pour nourriture
        elif self.squirrel_pos in self.traps:
            self.hunger -= 30
            self.traps.remove(self.squirrel_pos)
            reward -= 1.0  # Pénalité pour piège
        elif self.squirrel_pos == self.ruby_pos:
            self.has_ruby = True
            reward += 10.0
            self.done = True

        # Conditions de fin
        if self.hunger <= 0:
            reward -= 10.0
            self.done = True
        if self.squirrel_pos in self.predators:
            reward -= 10.0
            self.done = True

        # Bouger les prédateurs
        self.move_predators()

        return self.get_state(), reward, self.done

    def get_state(self):
        """Retourne l'état sous forme de tableau NumPy (10x10)"""
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)

        # Remplir avec les éléments
        for x, y in self.obstacles:
            state[x][y] = OBSTACLE
        for x, y in self.foods:
            state[x][y] = FOOD
        for x, y in self.traps:
            state[x][y] = TRAP
        for x, y in self.predators:
            state[x][y] = PREDATOR
        state[self.ruby_pos] = RUBY
        sx, sy = self.squirrel_pos
        state[sx][sy] = SQUIRREL

        return state

    def get_state_with_hunger(self):
        """Retourne l'état avec la faim (2 canaux : grille + faim)"""
        state = self.get_state().astype(np.float32) / 6.0
        hunger_grid = self.hunger / MAX_HUNGER
        return state, hunger_grid 

    def get_flat_state(self):
        """État plat pour un réseau de neurones simple"""
        state = self.get_state().flatten().astype(np.float32) / 6.0
        hunger_normalized = self.hunger / MAX_HUNGER
        return np.append(state, hunger_normalized)  # Shape: 101

    def get_action_space(self):
        """Espace d'actions (4 directions)"""
        return 4

    def get_state_shape(self, include_hunger=True):
        """Taille de l'état (pour un réseau de neurones)"""
        if include_hunger:
            return (2, self.grid_size, self.grid_size)  # Avec faim (canaux)
        else:
            return (self.grid_size, self.grid_size)  # Sans faim

    def get_grid_for_render(self):
        """Pour l'affichage dans Streamlit (avec emojis)"""
        emoji_map = {
                EMPTY: "🟫", OBSTACLE: "🪨", FOOD: "🍎",
                TRAP: "🧨", PREDATOR: "🐺", SQUIRREL: "🐿️", RUBY: "🍄"
        }
        display = np.full((self.grid_size, self.grid_size), emoji_map[EMPTY])

        # Remplir avec les éléments
        for x, y in self.obstacles:
            display[x][y] = emoji_map[OBSTACLE]
        for x, y in self.foods:
            display[x][y] = emoji_map[FOOD]
        for x, y in self.traps:
            display[x][y] = emoji_map[TRAP]
        for x, y in self.predators:
            display[x][y] = emoji_map[PREDATOR]
        rx, ry = self.ruby_pos
        display[rx][ry] = emoji_map[RUBY]
        sx, sy = self.squirrel_pos
        display[sx][sy] = emoji_map[SQUIRREL]

        return display