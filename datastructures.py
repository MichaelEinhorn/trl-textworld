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
        self.queue.clear()


class RejectionBuffer:
    def __init__(self, min=True, rank=0, world_size=1):
        self.values = []
        self.text = []
        self.min = min
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(zip(self.text, self.values))
    
    def __len__(self):
        return len(self.values)

    def append(self, t, v):
        self.values.append(v)
        self.text.append(t)

    def reject(self, p, threshType="frac"):
        # gather all data to rank 0 before rejecting
        if self.rank == 0:
            gathered_values = [None for i in range(self.world_size)]
            gathered_text = [None for i in range(self.world_size)]
        else:
            gathered_values = None
            gathered_text = None
        torch.distributed.gather_object(self.values, object_gather_list=gathered_values, dst=0)
        torch.distributed.gather_object(self.text, object_gather_list=gathered_text, dst=0)
        self.values = []
        self.text = []
        if self.rank == 0:
            for val, tex in zip(gathered_values, gathered_text):
                self.values.extend(val)
                self.text.extend(tex)

            if threshType == "frac":
                idxs = torch.argsort(torch.stack(self.values))
                if not self.min:
                    # idxs = idxs[::-1]
                    idxs = torch.flip(idxs, [0])

                num = len(self.values * p)
                idxs = idxs[:num]
                for i in range(num):
                    tempText.append(self.text[idxs[i]])
                    tempVal.append(self.values[idxs[i]])
                self.text = tempText
                self.values = tempVal

            if threshType == "top n":
                idxs = torch.argsort(torch.stack(self.values))
                if not self.min:
                    # idxs = idxs[::-1]
                    idxs = torch.flip(idxs, [0])
                idxs = idxs[:p]
                tempText = []
                tempVal = []
                for i in range(p):
                    tempText.append(self.text[idxs[i]])
                    tempVal.append(self.values[idxs[i]])
                self.text = tempText
                self.values = tempVal

    def clear(self):
        self.values = []
        self.text = []

    def sample(self, batch_size):
        # all samples are on rank 0, shuffle them there and then broadcast to other processes
        tmpArr = [None, None]
        if self.rank == 0:
            idxs = np.random.choice(len(self.values), batch_size * self.world_size, replace=False)
            tempText = []
            tempVal = []
            for i in range(batch_size * self.world_size):
                tempText.append(self.text[idxs[i]])
                tempVal.append(self.values[idxs[i]])
            tmpArr = [tempText, tempVal]

        torch.distributed.broadcast_object_list(tmpArr, src=0)

        tempText = tmpArr[0]
        tempVal = tmpArr[1]

        text = []
        val = []
        for i in range(self.rank, batch_size * self.world_size, self.world_size):
            text.append(tempText[i])
            val.append(tempVal[i])

        return text, val


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
        
    def clear(self):
        self.buffer.clear()

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
        scores, queries, responses, next_value, ret_cross, adv_cross, value, logprob = zip(*[self.buffer[idx] for idx in indices])

        return scores, queries, responses, next_value, ret_cross, adv_cross, value, logprob
        # return (np.array(scores, dtype=np.float32), np.array(queries), np.array(responses),
        #         np.array(values, dtype=np.float32), np.array(ret_cross, dtype=np.float32), np.array(adv_cross, dtype=np.float32))


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size: int = 200, rank=0, world_size=1):
        self.buffer = buffer
        self.sample_size = sample_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # objList = [None]
        # if self.rank == 0:
        #     data = self.buffer.sample(self.sample_size)
        #     objList = [data]
        #
        # torch.distributed.broadcast_object_list(objList, src=0)
        # data = objList[0]

        data = self.buffer.sample(self.sample_size)

        scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)
        for i in range(len(scores)):
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
        # print(lengths)
        # print(logprobs[0].shape, ref_logprobs[0].shape)
        # print(len(logprobs), logprobs[0].shape)
        # print(len(queries), queries[0].shape)
        
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

class DecisionDataset(IterableDataset):
    def __init__(self, buffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        input_ids = self.buffer.sample(self.sample_size)
        for i in range(len(input_ids)):
            yield input_ids[i]

class DecisionDatasetCollator():
    def __init__(self, text_collator=None):
        self.text_collator = text_collator

    def __call__(self, input_ids):
        # for inp in input_ids:
            # print(inp.shape)
        return self.text_collator(input_ids)

class LineBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def clear(self):
        self.buffer.clear()

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
    def __init__(self, buffer: RejectionBuffer, sample_size=200, rank=0, world_size=1):
        self.buffer = buffer
        self.sample_size = sample_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
#         objList = [None, None]
#         if self.rank == 0:
#             data, reject_scores = self.buffer.sample(self.sample_size)
#             objList = [data, reject_scores]
        
#         torch.distributed.broadcast_object_list(objList, src=0)
#         data = objList[0]
#         reject_scores = objList[1]

        data, reject_scores = self.buffer.sample(self.sample_size)

        scores, queries, responses, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward = zip(*data)
        for i in range(len(scores)):
            # padding matches the batches but this means padded querry + padded response is not padded (querry + response)
            model_input = torch.cat([queries[i], responses[i]])
            lengths = (queries[i].shape[0], responses[i].shape[0], model_input.shape[0])
            yield scores[i], queries[i], responses[i], model_input, lengths, values_next[i], ret_cross[i], adv_cross[i], logprobs[i], ref_logprobs[i], values[i], rewards[i], non_score_reward[i], reject_scores[i]

class RejectDatasetCollator():
    def __init__(self, text_collator=None):
        self.text_collator = text_collator

    def __call__(self, data):
        scores, queries, responses, model_input, lengths, values_next, ret_cross, adv_cross, logprobs, ref_logprobs, values, rewards, non_score_reward, reject_scores = zip(*data)
        # print(values)
        values = tuple([torch.squeeze(v, 1) for v in values])
        # print(lengths)
        # print(logprobs[0].shape, ref_logprobs[0].shape)
        # print(len(logprobs), logprobs[0].shape)
        # print(len(ref_logprobs), ref_logprobs[0].shape)
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
                padded_stack(non_score_reward),
                torch.tensor(reject_scores)
                )
