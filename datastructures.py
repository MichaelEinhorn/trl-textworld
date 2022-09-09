from collections import deque
import numpy as np
from torch.utils.data.dataset import IterableDataset
from typing import Iterable, Callable
import torch
from core import padded_stack


class RollingBuffer:
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


class RejectionBuffer:
    def __init__(self, min=True):
        self.values = []
        self.text = []
        self.min = min

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(zip(self.text, self.values))

    def append(self, t, v):
        self.values.append(v)
        self.text.append(t)

    def reject(self, p, threshType="frac"):
        if threshType == "frac":
            idxs = np.argsort(self.values)
            if not self.min:
                idxs = idxs[::-1]
            idxs = idxs[:len(self.values * p)]
            self.text = self.text[idxs]
            self.values = self.values[idxs]
        if threshType == "top n":
            idxs = np.argsort(self.values)
            if not self.min:
                idxs = idxs[::-1]
            idxs = idxs[:p]
            self.text = self.text[idxs]
            self.values = self.values[idxs]

    def clear(self):
        self.values = []
        self.text = []

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.values), batch_size, replace=False)
        return self.text[idxs], self.values[idxs]


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """
        Add experience to the buffer
        Args:
            experience: tuple (scores, queries, responses, values, ret_cross, adv_cross)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # original states, actions, rewards, dones, next_states
        scores, queries, responses, values, ret_cross, adv_cross = zip(*[self.buffer[idx] for idx in indices])

        return scores, queries, responses, values, ret_cross, adv_cross
        # return (np.array(scores, dtype=np.float32), np.array(queries), np.array(responses),
        #         np.array(values, dtype=np.float32), np.array(ret_cross, dtype=np.float32), np.array(adv_cross, dtype=np.float32))


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        data = self.buffer.sample(self.sample_size)
        scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)
        for i in range(0, len(scores)):
            # padding matches the batches but this means padded querry + padded response is not padded (querry + response)
            model_input = torch.cat([queries[i], responses[i]])
            lengths = (queries[i].shape[0], responses[i].shape[0], model_input.shape[0])
            yield scores[i], queries[i], responses[i], model_input, lengths, values_next[i], ret_cross[i], adv_cross[i], logprobs[i], ref_logprobs[i], values[i], rewards[i], non_score_reward[i]


class RLDatasetCollator():
    def __init__(self, text_collator=None):
        self.text_collator = text_collator

    def __call__(self, data):
        scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)
        # print(values)
        values = tuple([torch.squeeze(v, 1) for v in values])
        return (torch.tensor(scores),
                self.text_collator(queries),
                self.text_collator(responses),
                self.text_collator(model_input),
                torch.tensor(lengths),
                torch.tensor(values_next),
                torch.tensor(ret_cross),
                torch.tensor(adv_cross),
                padded_stack(logprobs),
                padded_stack(ref_logprobs),
                padded_stack(values),
                padded_stack(rewards),
                padded_stack(non_score_reward)
                )

class LineBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # original states, actions, rewards, dones, next_states
        lines = [self.buffer[idx] for idx in indices]

        return lines


class LineDataset(IterableDataset):
    def __init__(self, buffer: LineBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        lines = self.buffer.sample(self.sample_size)
        for i in range(len(lines)):
            yield lines[i]


class RejectDataset(IterableDataset):
    def __init__(self, buffer: RejectionBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        lines = self.buffer.sample(self.sample_size)
        for i in range(len(lines)):
            yield lines[i]
