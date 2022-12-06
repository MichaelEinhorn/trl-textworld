import sys
sys.path.append("..")

import torch
from datastructures import RejectionBuffer

def test_reject():
    global_rank = 0
    world_size = 1
    game_batch_size = 128
    batch_size = game_batch_size // 2

    arr = [i for i in range(game_batch_size)]
    arr = arr.shuffle()
    text_arr = [(str(i), "stuff") for i in arr]

    rb = RejectionBuffer(sortMax=True, rank=global_rank,
                                              world_size=world_size)

    for i in range(game_batch_size):
        rb.append(text_arr[i], arr[i])

    rb.reject(batch_size, threshType="top n")

    assert len(rb.values) == batch_size
    assert len(rb.text) == batch_size
    text, val = rb.sample(batch_size)
    for i in range(batch_size):
        idx = val[i]
        assert text[i] == (str(idx), "stuff")
