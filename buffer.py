import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = torch.zeros((self.mem_size, *input_shape))
        self.new_state_memory = torch.zeros((self.mem_size, *input_shape))
        self.action_memory = torch.zeros((self.mem_size, n_actions))
        self.reward_memory = torch.zeros(self.mem_size)
        self.done_memory = torch.zeros(self.mem_size, dtype = torch.bool)

    def store_trans(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        
        self.state_memory[index] = torch.tensor(state, dtype = torch.float32)
        self.new_state_memory[index] = torch.tensor(state_, dtype = torch.float32)
        self.action_memory[index] = torch.tensor(action, dtype = torch.float32)
        self.reward_memory[index] = torch.tensor(reward)
        self.done_memory[index] = torch.tensor(done)

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, states_, dones