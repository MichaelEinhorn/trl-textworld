from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from torch import nn
from copy import deepcopy
import os
from functools import reduce
from itertools import chain
import deepspeed


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""

    def __init__(self, n_embd=700, n_out=1, detach_head=False, layers=1, hidden_scale=2):
        super().__init__()
            
        self.detach_head = detach_head

        if layers == 1: # from trl
            self.summary = nn.Linear(n_embd, n_out)
        elif layers == 2: # from trlx make head
            self.summary = nn.Sequential(nn.Linear(n_embd, int(n_embd * hidden_scale)), nn.ReLU(), nn.Linear(int(n_embd * hidden_scale), n_out))
        else:
            raise NotImplementedError("Only 1 or 2 layers are supported for the value head.")

    def forward(self, hidden_states):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states

        # print("v head forward ", output.shape)
        output = self.summary(output)
        return output