import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time
import argparse
import sys

from gym_torcs1 import TorcsEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

state_dim = 10
action_dim = 3
max_action = None
Transition = namedtuple('Transition', ['s','a','r','s_','d'])
learning_rate = 0.0001
capacity = 400
gradient_steps = 20
batch_size = 64
gamma = 0.99
tau = 0.001
steer_range = (-1,1)
throttle_range = (0.8,1.0)
break_range = (0.0,0.5)
min_Val = torch.tensor(1e-7).float().to(device)
total_episodes = 1000
max_episode_length = 400
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=action_dim, min_log_std=-20, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,512)
        self.mu_head = nn.Linear(512,action_dim)
        self.log_std_head = nn.Linear(512,action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        # print(s.shape,type(a))
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self):
        super(SAC, self).__init__()

        self.policy_net = Actor(state_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=learning_rate)

        self.replay_buffer = [Transition] * capacity
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 0

        self.writer = SummaryWriter('./test_agent')



        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        # print(action,action1[0],action1[1],action1[2])
        action[0] = (action[0] + 1)/2 * (steer_range[1] - steer_range[0]) + steer_range[0]
        action[1] = (action[1] + 1)/2 * (throttle_range[1] - throttle_range[0]) + throttle_range[0]
        action[2] = (action[2] + 1)/2 * (break_range[1] - break_range[0]) + break_range[0]

        return action # return a scalar, float32

    def store(self, s, a, r, s_, d):
        index = self.num_transition % capacity
        transition = Transition(s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma*z.to(device))
        # print("LOGG",torch.sum(dist.log_prob(batch_mu + batch_sigma * z.to(device)),dim=1))
        # print(torch.sum(torch.log(torch.clamp((1 - action.pow(2)),min=0,max=1) + min_Val), dim=1))
        log_prob = torch.sum(dist.log_prob(batch_mu + batch_sigma * z.to(device)),dim=1) - torch.sum(torch.log(torch.clamp((1 - action.pow(2)),min=0,max=1) + min_Val), dim=1)
        log_prob1 = log_prob.view(batch_size,1)
        # print(log_prob.shape)
        return action, log_prob1, z, batch_mu, batch_log_sigma


    def update(self,s,a,r,s_,d):
        if self.num_training % 500 == 0:
            print("Training ... \t{} times ".format(self.num_training))
        # print(self.replay_buffer[0])
        # for _ in range(gradient_steps):
        for index in BatchSampler(SubsetRandomSampler(range(capacity)), batch_size, False):
            # index = np.random.choice(range(capacity), batch_size, replace=False)
            bn_s = s[index].reshape(-1, state_dim)
            bn_a = a[index].reshape(-1, action_dim)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index].reshape(-1, state_dim)
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            next_value = excepted_new_Q - log_prob
            # print(excepted_Q1.shape,next_q_value.detach().shape,excepted_Q2.shape)

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            pi_loss = (log_prob - excepted_new_Q).mean() # according to original paper
            # print(log_prob.shape, excepted_new_Q.shape)

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            print("LOSSES(V, Q1, Q2, pi)",V_loss.item(),Q1_loss.item(),Q2_loss.item(),pi_loss.item())
            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - tau) + param * tau)

            self.num_training += 1


    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net1.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load( './SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        print("model has been loaded")

if __name__ == "__main__":
    sac = SAC()
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    for episode in range(total_episodes):

        if np.mod(episode,3) == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        done = False

        episode_reward = 0.0
        episode_length = 0

        for j in range(max_episode_length):
            print("START")

            if(episode>0):
                a=sac.select_action(s)
            else:
                a = np.array([0.0,1.0,0.0])
            print("ACTING")
            ob, r, done, _ = env.step(a)
            print("REWARDD", r)
            s2 = np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            episode_reward += r
            episode_length += 1

            sac.store(s,a,r,s2,float(done))
            s=s2
            if sac.num_transition > capacity:
                s11 = torch.tensor([t.s for t in sac.replay_buffer]).float().to(device)
                a = torch.tensor([t.a for t in sac.replay_buffer]).float().to(device)
                r = torch.tensor([t.r for t in sac.replay_buffer]).float().to(device)
                s_ = torch.tensor([t.s_ for t in sac.replay_buffer]).float().to(device)
                d = torch.tensor([t.d for t in sac.replay_buffer]).float().to(device)
                sac.update(s11,a,r,s_,d)
                # ob, r, done, _ = env.step(a)
            if done:
                break
        if episode % 10 == 0:
            sac.save()
        sac.writer.add_scalar('ep_r', episode_reward, global_step=episode)
        print("EPISODE {} with REWARD {}".format(episode, episode_reward))
    env.end()
