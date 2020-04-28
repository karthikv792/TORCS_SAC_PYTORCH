import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from gym_torcs1 import TorcsEnv

import matplotlib.pyplot as plt


#Use CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def plot(frame_idx, rewards):
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('Episode %s. Reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.savefig('plot.png', dpi=300, bbox_inches='tight')
	plt.close()

def plot_rewards(partial_rewards_list):
	r1 = [i['Reward1'] for i in partial_rewards_list]
	r2 = [i['Reward2'] for i in partial_rewards_list]
	r3 = [i['Reward3'] for i in partial_rewards_list]

	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('Reward based on Following Trajectory')
	plt.plot(r1)
	plt.savefig('r1.png', dpi=300, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('Reward based on Following Trajectory')
	plt.plot(r2)
	plt.savefig('r2.png', dpi=300, bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('Reward based on Following Trajectory')
	plt.plot(r3)
	plt.savefig('r3.png', dpi=300, bbox_inches='tight')
	plt.close()


#Replay Buffer

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done
	def __len__(self):
		return len(self.buffer)

class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim, init_w=3e-3):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class SoftQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3):
		super(SoftQNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs+num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w,init_w)

	def forward(self, state, action):
		x = torch.cat([state,action],1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3, log_std_min = -20, log_std_max = 2):
		super(PolicyNetwork, self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)
		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		mean    = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std

	def evaluate(self, state, epsilon=1e-4):
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean,std)
		z = normal.sample()
		action = torch.tanh(z)

		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)
		# print(log_prob.shape)

		return action, log_prob, z, mean, log_std

	def get_action(self, state):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean, std)
		z  = normal.sample()
		action = torch.tanh(z)
		action = action.detach().cpu().numpy()

		action[0][0] = (action[0][0]+1)*0.5*(steer_range[1]-steer_range[0]) + steer_range[0]
		action[0][1] = (action[0][1]+1)*0.5*(accel_range[1]-accel_range[0]) + accel_range[0]
		action[0][2] = (action[0][2]+1)*0.5*(brake_range[1]-brake_range[0]) + brake_range[0]
		# print(action)
		return action[0]

def softQUpdate(batch_size, gamma=0.99,mean_lambda=1e-3,std_lambda=1e-3,z_lambda=0.0,soft_tau=1e-2):
	state, action, reward, next_state, done= replay_buffer.sample(batch_size)

	state = torch.FloatTensor(state).to(device)
	next_state = torch.FloatTensor(next_state).to(device)
	action = torch.FloatTensor(action).to(device)
	reward = torch.FloatTensor(reward).to(device)
	done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

	expected_q_value = soft_q_net(state,action)
	expected_value = value_net(state)

	new_action,log_prob,z,mean,log_std = policy_net.evaluate(state)

	target_value = target_value_net(next_state)
	next_q_value = reward.view(len(reward),1) + (1-done) * gamma * target_value
	# print("SHAPE1", expected_q_value.shape, reward.shape,expected_value.shape)
	q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

	expected_new_q_value = soft_q_net(state, new_action)
	next_value = expected_new_q_value - log_prob
	value_loss = value_criterion(expected_value, next_value.detach())

	log_prob_target = expected_new_q_value - expected_value
	policy_loss =  (log_prob * (log_prob - log_prob_target).detach()).mean()

	mean_loss = mean_lambda * mean.pow(2).mean()
	std_loss = std_lambda * log_std.pow(2).mean()
	z_loss = z_lambda * z.pow(2).sum(1).mean()

	policy_loss += mean_loss + std_loss + z_loss

	soft_q_optimizer.zero_grad()
	q_value_loss.backward()
	soft_q_optimizer.step()

	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()

	policy_optimizer.zero_grad()
	policy_loss.backward()
	policy_optimizer.step()

	for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
		target_param.data.copy_( target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


def slip(ob):
	deg2 = -ob.angle
	sp = ob.speedX
	spy = ob.speedY
	x = sp*np.cos(deg2) - spy*np.sin(deg2)
	y = spy*np.cos(deg2) + sp*np.sin(deg2)
	slip = (math.atan(y/x)/math.pi)*180
	return slip
def save():
    torch.save(policy_net.state_dict(), './SAC_model/policy_net.pth')
    torch.save(value_net.state_dict(), './SAC_model/value_net.pth')
    torch.save(soft_q_net.state_dict(), './SAC_model/Q_net.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

def load():
    policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
    value_net.load_state_dict(torch.load( './SAC_model/value_net.pth'))
    soft_q_net.load_state_dict(torch.load('./SAC_model/Q_net.pth'))
    print("model has been loaded")

def action_smoothing(action,prev_action):
	#Action Smoothing
	smooth_factor_curr = [0.1,0.3,0.0]
	smooth_factor_prev = [0.9,0.7,0.0]
	new_action = []
	new_action.append(action[0]*smooth_factor_curr[0] + prev_action[0]*smooth_factor_prev[0])
	new_action.append(action[1]*smooth_factor_curr[1] + prev_action[1]*smooth_factor_prev[1])
	new_action.append(action[2]*smooth_factor_curr[2] + prev_action[2]*smooth_factor_prev[2])
	print(new_action)
	return new_action
env = TorcsEnv(vision = False, throttle=True, gear_change=False)
action_dim = 3
state_dim = 29
hidden_dim = 256
steer_range = (-0.8,0.8)
accel_range = (0.6,1)
brake_range = (0,0)
value_net = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
load()
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
	target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
soft_q_criterion = nn.MSELoss()
value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 4000000
max_steps = 2000000
frame_idx = 0
rewards = []
batch_size = 128
episode=0
partial_rewards_list = []
ob = env.reset(relaunch=True)
state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
episode_reward = 0
prev_action = [0.0,1.0,0.0]
for steps in range(max_steps):
	action = policy_net.get_action(state)
	ob, reward, done, _ = env.step(action)
		# partial_rewards_list.append(partial_rewards)
	next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
	state= next_state
	prev_action = action
	episode_reward+=reward

		# if frame_idx%100==0:
		# 	plot_rewards(partial_rewards_list)
	if done:
		break
rewards.append(episode_reward)
print("EPISODE %s with Reward %s" %(episode, episode_reward))
