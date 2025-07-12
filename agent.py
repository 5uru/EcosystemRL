import copy
from tinygrad import Tensor, nn, dtypes
import math
import random
import tqdm
from ruby_quest import RubyQuestRL
from utils import ReplayMemory
from model import AgentModel
import matplotlib.pyplot as plt
from collections import deque

SEED = 1234
CAPACITY = 5000
BATCH_SIZE = 128
N_ACTIONS = 4
LEARNING_START = CAPACITY
UPDATE_FREQ = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_STEPS = 30_000
GAMMA = 0.99
TARGET_UPDATE = 1_000
PRINT_UPDATE = 5_000
MAX_STEPS = 200_000 # Define a maximum number of steps to ensure training terminates

class Agent:
    def __init__(self, env, mem, model, update_freq, learning_start, e_start, e_end, e_steps, gamma, target_update, print_update):
        """
        An agent class that handles training the model

        args:
            mem (ReplayMemory): ReplayMemory object
            env (Environment): Environment object
            model (nn.Module): PyTorch model
            update_freq (int): we only update the model every update_freq steps, 1 means update every step
            learning_start (int): we only start updating the model after learning_start steps
            e_start (int): initial value of epsilon
            e_end (int): minimum value of epsilon
            e_steps (int): controls the rate of decay from e_start to e_end
            gamma (float): decay rate of rewards
            target_update (int): update target model after this many parameter updates
            print_update (int): print summary of performance after this many steps
        """

        self.env = env
        self.mem = mem
        self.model = model
        self.update_freq = update_freq
        self.learning_start = learning_start
        self.e_start = e_start
        self.e_end = e_end
        self.e_steps = e_steps
        self.gamma = gamma
        self.target_update = target_update
        self.print_update = print_update

        self.steps = 0 #number of steps taken
        self.episodes = 0 #number of episodes
        self.param_updates = 500 #number of parameter updates

        #create target model
        self.target = copy.deepcopy(self.model)

        #create optimizer
        self.optimizer = nn.optim.Adam(nn.state.get_parameters(self.model), lr=1e-4)

    def get_epsilon(self):
        """
        Calculates the value of epsilon from the current number of steps

        returns:
            epsilon (float): the probability of doing a random action
        """
        if self.steps > self.e_steps:
            return self.e_end
        epsilon = self.e_end + (self.e_start - self.e_end) * math.exp(-1. * self.steps / self.e_steps)
        return epsilon

    def get_action(self, state):
        """
        Selects action to perform, with probability = epsilon chooses a random action,
        else chooses the best predicted action of the model

        args:
            state (tuple): input state to the model

        returns:
            action (int): the index of the action
        """
        epsilon = self.get_epsilon()

        if random.random() < epsilon:
            action = Tensor.randint(1, low=0, high=N_ACTIONS)[0]
        else:
            Tensor.training = False
            q_values = self.model(state)
            action = q_values.argmax()

        return action

    def train(self):
        """
        Main training loop of the model.
        """
        all_episode_rewards = []
        rewards_deque = deque(maxlen=100)
        pbar = tqdm.tqdm(total=MAX_STEPS)

        while self.steps < MAX_STEPS:
            self.episodes += 1
            episode_reward = 0.0
            episode_done = False

            self.env.reset()
            state_array, hunger = self.env.get_state_with_hunger()
            state_array = state_array.reshape(1, 1, *state_array.shape)
            state = (Tensor(state_array), Tensor([hunger]))

            while not episode_done:
                pbar.update(1)
                action = self.get_action(state)

                _, reward, episode_done = self.env.step(action.numpy())
                episode_reward += reward

                next_state = None
                if not episode_done:
                    next_state_array, next_hunger = self.env.get_state_with_hunger()
                    next_state_array = next_state_array.reshape(1, 1, *next_state_array.shape)
                    next_state = (Tensor(next_state_array), Tensor([next_hunger]))

                self.mem.push(state, action, reward, next_state)
                state = next_state
                self.steps += 1

                if self.steps > self.learning_start and self.steps % self.update_freq == 0:
                    loss = self.optimize()

                if self.steps > self.learning_start and self.steps % self.target_update == 0:
                    self.target = copy.deepcopy(self.model)
                if self.steps % self.print_update == 0:
                    avg_reward = sum(rewards_deque) / len(rewards_deque) if rewards_deque else 0.0
                    print(f'\nEpisodes: {self.episodes}, Steps: {self.steps}, Epsilon: {self.get_epsilon():.2f}, Avg. Reward (Last 100): {avg_reward:.2f}')

                if self.steps >= MAX_STEPS:
                    episode_done = True # Exit inner loop

            all_episode_rewards.append(episode_reward)
            rewards_deque.append(episode_reward)

        pbar.close()
        return all_episode_rewards

    def optimize(self):

        if len(self.mem) < self.mem.batch_size:
            return Tensor(0.0)

        transitions = self.mem.sample()

        # Create non-terminal mask (1 for non-terminal, 0 for terminal)
        non_terminal_mask = Tensor([s is not None for s in transitions.next_state])

        # Process states - separate spatial and hunger components
        spatial_states = [s[0] for s in transitions.state]
        state_spatial_batch = Tensor.cat(*spatial_states, dim=0)

        hunger_states = [s[1] for s in transitions.state]
        state_hunger_batch = Tensor.cat(*hunger_states, dim=0)

        # Process actions and rewards
        action_batch = Tensor([a.numpy() for a in transitions.action], dtype=dtypes.int32).reshape(self.mem.batch_size, 1)
        reward_batch = Tensor(list(transitions.reward)).reshape(self.mem.batch_size, 1)



        # Initialize target values as zeros
        target_vals = Tensor.zeros(self.mem.batch_size, 1)

        Tensor.training = False
        # Process non-terminal next states
        if any(non_terminal_mask.numpy()):
            next_spatial_states = [s[0] for s in transitions.next_state if s is not None]
            next_hunger_states = [s[1] for s in transitions.next_state if s is not None]

            next_state_spatial_batch = Tensor.cat(*next_spatial_states, dim=0)
            next_state_hunger_batch = Tensor.cat(*next_hunger_states, dim=0)

            # Get target predictions for non-terminal next states
            target_pred = self.target((next_state_spatial_batch, next_state_hunger_batch))
            max_target_vals = target_pred.max(1)[0].reshape(-1, 1)

            # Fill in target values only for non-terminal states
            target_vals = target_vals.where(non_terminal_mask.reshape(-1, 1) == 0, max_target_vals)
        Tensor.training = True
        # Calculate expected Q values
        expected_vals = reward_batch + (target_vals * self.gamma)

        self.optimizer.zero_grad()
        # Get Q predictions for current states
        q_preds = self.model((state_spatial_batch, state_hunger_batch))
        q_vals = q_preds.gather(1, action_batch)

        # Calculate loss
        loss = (q_vals - expected_vals.detach()).pow(2).mean()

        # Optimize

        loss.backward()
        self.optimizer.step()

        return loss

def plot_rewards(rewards):
    """Plots total rewards per episode and a 100-episode moving average."""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward per Episode', alpha=0.6)

    moving_avg = [sum(rewards[max(0, i-100):i+1]) / len(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
    plt.plot(moving_avg, label='100-Episode Moving Average', color='red', linewidth=2)

    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('reward_plot.png')
    print("Reward plot saved to 'reward_plot.png'")
    plt.show()

if __name__ == "__main__":
    env = RubyQuestRL()
    mem = ReplayMemory(CAPACITY, BATCH_SIZE)
    model = AgentModel()
    agent = Agent(env, mem, model, UPDATE_FREQ, LEARNING_START, EPSILON_START, EPSILON_END, EPSILON_STEPS, GAMMA, TARGET_UPDATE, PRINT_UPDATE)

    print("Starting training...")
    episode_rewards = agent.train()
    print("Training completed.")

    # Plot and save the reward variation
    if episode_rewards:
        plot_rewards(episode_rewards)