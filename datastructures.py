from collections import deque
import numpy as np
from torch.utils.data.dataset import IterableDataset


class buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque([])

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def append(self, obj):
        self.queue.append(obj)
        if len(self.queue) > self.max_size:
            return self.queue.pop()
        else:
            return None

    def clear(self):
        self.queue = deque([])


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
