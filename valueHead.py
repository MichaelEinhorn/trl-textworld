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

# https://github.com/CarperAI/trlx/blob/main/trlx/model/nn/ilql_models.py
class ILQLHeads(nn.Module):
    def __init__(self, n_embd=700, n_out=50000, detach_head=False, layers=1, hidden_scale=2, n_qs=2, alpha=1.0, targetHead=False, zero3=False):
        super().__init__()

        self.n_embd = n_embd
        self.n_out = n_out
        self.v_head = ValueHead(n_embd, n_out=1, detach_head=detach_head, layers=layers, hidden_scale=hidden_scale)
        self.zero3 = zero3
        self.alpha = alpha
        self.hasTargetHead = targetHead

        self.q_heads = nn.ModuleList(
            ValueHead(self.n_embd, self.n_out, detach_head=detach_head, layers=layers, hidden_scale=hidden_scale) for _ in range(n_qs)
        )
        if self.hasTargetHead:
            self.target_q_heads = nn.ModuleList(
                ValueHead(self.n_embd, self.n_out, detach_head=detach_head, layers=layers, hidden_scale=hidden_scale) for _ in range(n_qs)
            )
            self.sync_target_q_heads()

            for q_head in self.target_q_heads:
                q_head.requires_grad_(False)

    def forward(self, hidden_states, states_ixs=None, actions_ixs=None):
        output = {}
        if states_ixs is not None:
            states_hs = hidden_states.gather(
                dim=1, index=states_ixs.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
            )
            actions_hs = hidden_states.gather(
                dim=1, index=actions_ixs.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
            )
        else:
            states_hs = actions_hs = hidden_states

        # print("q forward", hidden_states.shape)

        qs = tuple(q_head(actions_hs) for q_head in self.q_heads)
        vs = self.v_head(states_hs)
        output["qs"] = qs
        output["vs"] = vs

        if self.hasTargetHead:
            target_qs = tuple(q_head(actions_hs) for q_head in self.target_q_heads)
            output["target_qs"] = target_qs
        
        return output

    def _sync_target_q_heads(self, alpha):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(
                target_q_head.parameters(), q_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        if not self.hasTargetHead:
            return

        if self.zero3:
            params = chain(
                chain(q_head.parameters() for q_head in self.q_heads),
                chain(q_head.parameters() for q_head in self.target_q_heads),
            )

            with deepspeed.zero.GatheredParameters(list(params), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    self._sync_target_q_heads(self.alpha)
        else:
            self._sync_target_q_heads(self.alpha)
