import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)  # event shape (state, action, ...)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(
            *random.sample(self.memory, batch_size)
        )  # samples shape: [(states), (actions)]
        # samples = zip(*self.memory[:batch_size])
        # return map(lambda x: torch.cat(x, 0), samples)
        return map(lambda x: np.array(x), samples)

    def sample_cuda(self, batch_size):
        samples = zip(
            *random.sample(self.memory, batch_size)
        )  # samples shape: [(states), (actions)]
        return map(lambda x: torch.cat(x, 0), samples)

    def pop(self, batch_size):
        mini_batch = zip(*self.memory[:batch_size])
        # return map(lambda x: torch.cat(x, 0), mini_batch)
        return map(lambda x: np.array(x), mini_batch)

    def return_size(self):
        return len(self.memory)
