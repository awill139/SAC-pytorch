import os
import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from network import Actor, Critic, Value

class Agent():
    def __init__(self, alpha = 0.0003, beta = 0.0003, input_dims = [8],
                 env = None, gamma = 0.99, tau = 0.005, n_actions = 2, max_size = 1000000,
                 layer1_size = 256, layer2_size = 256, batch_size = 256, reward_scale = 2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        self.memory = ReplayBuffer(max_size, input_dims, n_actions = n_actions)
        self.actor = Actor(alpha, input_dims, n_actions = n_actions, max_action = env.action_space.n)
        self.critic1 = Critic(beta, input_dims, n_actions = n_actions, name = 'critic1')
        self.critic2 = Critic(beta, input_dims, n_actions = n_actions, name = 'critic2')
        self.value = Value(beta, input_dims, name = 'value')
        self.target_value = Value(beta, input_dims, name = 'target')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.update_network_params(tau = 1)

    def choose_action(self, obs):
        state = torch.tensor([obs],dtype=torch.float32).to(self.device)
        actions, _ = self.actor.sample_normal(state, reparam = False)

        return actions.cpu().detach().numpy()[0]

    def store_trans(self, state, action, reward, new_state, done):
        self.memory.store_trans(state, action, reward, new_state, done)

    def update_network_params(self, tau = None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict.keys():
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1 - tau) * target_value_state_dict[name].clone()

        def save_models(self):
            self.actor.save_checkpoint()
            self.value.save_checkpoint()
            self.target_value.save_checkpoint()
            self.critic1.save_checkpoint()
            self.critic2.save_checkpoint()
            print('saving models')
        def load_models(self):
            self.actor.load_checkpoint()
            self.value.load_checkpoint()
            self.target_value.load_checkpoint()
            self.critic1.load_checkpoint()
            self.critic2.load_checkpoint()
            print('loading models')

    def get_critic_val_log_prob(self, state, reparam):
        actions, log_probs = self.actor.sample_normal(state, reparam = False)
        log_probs = log_probs.view(-1)
        q1_new = self.critic1(state, actions)
        q2_new = self.critic2(state, actions)
        critic_value = torch.min(q1_new, q2_new)
        critic_value = critic_value.view(-1)

        return log_probs, critic_value

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        state_ = torch.tensor(new_state,dtype=torch.float32).to(self.device)
        done = torch.tensor(done).to(self.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value[done] = 0.0

        critic_value, log_probs = self.get_critic_val_log_prob(state, reparam = False)

        self.value.optimizer.zero_grad()

        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph = True)
        self.value.optimizer.step()

        critic_value, log_probs = self.get_critic_val_log_prob(state, reparam = True)

        self.actor.optimizer.zero_grad()

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        q_hat = self.scale * reward + self.gamma * value_
        q1_old = self.critic1(state, action).view(-1)
        q2_old = self.critic2(state, action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_params()