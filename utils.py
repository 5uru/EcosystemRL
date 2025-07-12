from collections import deque
from collections import namedtuple
from tinygrad import Tensor
import random

class ReplayMemory:
    def __init__(self, capacity, batch_size):
        """
        Replay memory that holds examples in the form of (s, a, r, s')

        args:
            capacity (int): the size of the memory
            batch_size (int): size of batches used for training model
        """

        self.batch_size = batch_size
        self.capacity = capacity

        #the memory holds al the (s, a, r, s') pairs
        #a deque is first-in-first-out, i.e. when you push an example onto the queue
        #and it at maximum capacity, the oldest example is popped off the queue
        self.memory = deque(maxlen=self.capacity)

        #examples in the queue are saved as Transitions
        #makes the code more readable and intuitive when getting  examples from the memory
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def push(self, state, action, reward, next_state):
        """
        Places an (s, a, r, s') example in the memory

        args:
            state (np.array): the observation obtained from the environment before the action
            action (list[int]): the action taken
            reward (list[int]): the reward received from the action taken at the current state
            next_state (np.array or None): the observation obtained from the environment after the action,
                                           is None when the state is a terminal state
        """



        #create a transition
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)

        #add to the memory
        self.memory.append(transition)

    def sample(self):
        """
        Gets a random sample of n = batch_size examples from the memory
        The transition returned contains n of each elements, i.e. a batch_size of 32
        means this will return a tuple of (32 states, 32 actions, 32 rewards, 32 next_states)

        returns:
            Transitions (namedtuple): a tuple of (s, a, r, s'),
        """

        #sample batch_size transitions for the memory
        transitions = random.sample(self.memory, self.batch_size)

        #unzip and then rezip so each element contains batch_size examples
        return self.Transition(*(zip(*transitions)))

    def __len__(self):
        """
        Returns the length of the memory, i.e. number of examples in the memory

        returns:
            length (int): number of examples in the memory
        """

        return len(self.memory)


if __name__ == "__main__":
    # Example usage
    memory = ReplayMemory(capacity=1000, batch_size=2)

    # Push some dummy data
    for _ in range(10):
        state = Tensor.randn(1, 10, 10)  # Example state
        action = Tensor.randint(0, 3)  # Random action
        reward = Tensor.randint()  # Random reward
        next_state = Tensor.randn(1, 10, 10) if random.random() > 0.5 else None  # Random next state or None
        memory.push(state, action, reward, next_state)

    print(f"Memory size: {len(memory)}")
    sample = memory.sample()
    print(f"Sampled actions: {sample.action[0].numpy()}")
    print(f"Sampled rewards: {sample.reward[0].numpy()}")