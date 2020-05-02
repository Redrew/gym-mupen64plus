import gym_mupen64plus
# %%
import gym, torch, cv2, time
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import deque
import pickle
from os import path

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
version = input("Version: ")

# %%
_ACTION_DIM = 15

def process_obs(image):
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

# linearly decays epsilon
def epsilon_scheduler(args):
  epsilon = max(0, args.epsilon - args.iteration * args.epsilon_decay)
  return epsilon

def map_action_space(action_i):
    action = [0] * 8
    if action_i == 0:
        pass
    elif action_i == 1:
        action[0] = 127
    elif action_i == 2:
        action[0] = -128
    elif action_i == 3:
        action[1] = 127
    elif action_i == 4:
        action[1] = -128
    elif action_i == 5:
        action[0] = 127
        action[1] = 127
    elif action_i == 6:
        action[0] = 127
        action[1] = -128
    elif action_i == 7:
        action[0] = -128
        action[1] = 127
    elif action_i == 8:
        action[0] = -128
        action[1] = -128
    else:
        action[action_i - 7] = 1
    return action

def log(args, update=False):
    if update:
        args.losses.append(args.loss)
        if args.iteration % 10 == 0:
            print(args.loss)

# %%
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = True)
        
        return [self.buffer[i] for i in index]

class FrameStack:
    def __init__(self, stack_size):
        self.frames = deque(maxlen=stack_size)

    def __call__(self, frame):
        if len(self.frames) == 0:
            self.frames.extend([frame] * 4)
        else:
            self.frames.append(frame)
        stack = np.stack(self.frames, axis=0)
        return stack

class SmashEnv:
    def __init__(self, args):
        self.env = gym.make(args.env_id)
        self.args = args
    
    def step(self, action):
        if "Smash" in args.env_id:
            action = map_action_space(action)
            
        obs, reward, done, info = self.env.step(action)
        state = self.args.framestack(process_obs(obs))
        return state, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        state = self.args.framestack(process_obs(obs))
        return state
    
    def close(self): self.env.close()


def get_action(args, state):
    epsilon = epsilon_scheduler(args)
    args.model.eval()

    if np.random.rand() < epsilon:
        action = np.random.randint(args.action_dim)
    else:
        state_t = torch.Tensor(state).unsqueeze(0).to(device)
        Q = args.model(state_t)
        action = torch.max(Q, 1)[1].item()
    return action

def get_model(input_dim, output_dim):
    return nn.Sequential(
        # 64, 64
        nn.Conv2d(input_dim, 32, 7, 2, 3), # 32, 32
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), # 16, 16
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1), # 8, 8
        nn.ReLU(),
        nn.Flatten(), # 128 * 8 * 8
        nn.Linear(128 * 8 * 8, 1000),
        nn.ReLU(),
        nn.Linear(1000, output_dim)
    )

def get_loss(args):
    batch = args.memory.sample(args.batch_size)
    states, actions, rewards, next_states, dones = list(zip(*batch))
    states_t = torch.Tensor(states).to(device)
    next_states_t = torch.Tensor(next_states).to(device)

    Qs = args.model(states_t)
    next_Qs = args.target_model(next_states_t).detach()

    preds_t = Qs[np.arange(args.batch_size), actions]

    targets = []
    for i in range(args.batch_size):
        target = rewards[i]
        if not dones[i]:
            target += args.gamma * next_Qs[i].max()
        targets.append(target)
    targets_t = torch.Tensor(targets).to(device)
    
    loss = args.loss_func(preds_t, targets_t)
    
    args.loss = loss.item(); args.targets = targets_t; args.preds = preds_t
    
    return loss

def update(args):
    args.model.train()
    args.optimizer.zero_grad()

    loss = get_loss(args)
    loss.backward()

    if args.iteration % args.prop_steps == 0:
        args.target_model.load_state_dict(args.model.state_dict())

    log(args, update=True)
    args.iteration += 1

def init_env(args):
    args.env = SmashEnv(args)
    args.memory = Memory(args.memory_size)
    args.model = get_model(args.obs_dim, args.action_dim).to(device)
    args.target_model = get_model(args.obs_dim, args.action_dim).to(device)
    args.target_model.load_state_dict(args.model.state_dict())
    args.optimizer = torch.optim.Adam(args.model.parameters(), args.lr)


# %%
class Args: 
    def __init__(self):
        self.env_id = "Smash-dk-v0"

        self.nb_stacks = 4
        self.obs_dim = self.nb_stacks
        self.action_dim = _ACTION_DIM

        self.loss_func = nn.MSELoss()
        self.framestack = FrameStack(self.nb_stacks)

        self.epsilon = 0.1
        self.epsilon_decay = 0.000001
        self.gamma = 0.99
        self.lr = 0.003
        self.batch_size = 128
        self.prop_steps = 50

        self.memory_size = 100000
        self.nb_episodes = 1
        self.max_steps = 2000

        self.iteration = 0
        self.losses = []
        self.actions = []
        self.rewards = []
        self.targets = None
        self.preds = None

args = Args()


init_env(args)

# %%
env = args.env
for episode_idx in range(args.nb_episodes):
    state = env.reset()

    for step_idx in range(args.max_steps):
        last_state = state
        action = get_action(args, last_state)

        state, reward, done, _ = env.step(action)
        state = state

        args.memory.add((last_state, action, reward, state, done))
        
        update(args)


# %%







'''
def save_video(buffer, epoch):
  deep_frames = []
  for f in buffer.records[-args.max_steps:]:
    deep_frames += [f.state[0].T]
  plt.figure(figsize=(deep_frames[0].shape[1] / 72.0, deep_frames[0].shape[0] / 72.0), dpi = 72)                                          
  patch = plt.imshow(deep_frames[0])
  plt.axis('off')
  animate = lambda i: patch.set_data(deep_frames[i])
  ani = animation.FuncAnimation(plt.gcf(), animate, frames=len
  (deep_frames), interval = 50)

  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  ani.save('training%s_%i.mp4' % (version, epoch), writer=writer)

def save_training(model, buffer):
    torch.save(model.state_dict(), './DQN%s.pth' % version)
    #with open("buffer.pyobj", "w") as f:
    #    pickle.dump(buffer, f)

def load_training(model):
    if path.exists("./DQN%s.pth" % version):
      model.load_state_dict(torch.load("./DQN%s.pth" % version))
    #with open("buffer.pyobj", "r") as f:
    #    buffer = pickle.load(f)
    return buffer

def save_stats(args):
    with open("statistics%s.pyobj" % version, "a") as f:
        pickle.dump(args, f)


# %%
Feedback = namedtuple('feedback', ['state', 'reward', 'done', 'info'])
Record = namedtuple('record', ['action', 'state', 'reward', 'done'])
Transition = namedtuple('transition', ['state', 'action', 'next_state', 'reward', 'done'])

class ExpBuffer:
  def __init__(self, max_size = 30000):
    self.max_size = max_size
    self.records = []
    self.state_shape = None
  
  def add_record(self, action, state, reward, done):
    if len(self.records) == self.max_size: self.records.pop(0)
    self.state_shape = state.shape
    self.records.append(Record(action, state, reward, done))
  
  def add_state(self, state):
    self.add_record(None, state, None, False)

  def sample(self, batch_size, device='cpu'):
    if len(self.records) <= 1: raise Error('Sampling before buffer is filled') 
    states = torch.zeros(batch_size, *buffer.state_shape)
    actions = torch.zeros(batch_size, dtype=torch.long)
    next_states = torch.zeros(batch_size, *buffer.state_shape)
    rewards = torch.zeros(batch_size)
    done = torch.zeros(batch_size, dtype=torch.bool)

    for i in range(batch_size):
      while True:
        idx = np.random.randint(0, len(self.records))
        if idx != 0 and self.records[idx].action is not None: break
      record = self.records[idx]
      states[i] = self.records[idx-1].state
      actions[i] = record.action
      rewards[i] = record.reward
      if record.done: done[i] = True
      else: next_states[i] = record.state

    return Transition(states.to(device), actions.to(device), 
                      next_states.to(device), rewards.to(device), 
                      done.to(device))
    
    
    return Transition(self.records[idx-1].state, *self.records[idx])
    
  
  def __len__(self): return len(self.records)


def get_DQN(action_dim): 
  if version == 3:
    return nn.Sequential(
      nn.Conv2d(3, 8, 3, 1, 1), # 64, 64
      nn.ReLU(),
      nn.Conv2d(8, 16, 3, 2, 1), # 32, 32
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, 2, 1), # 16, 16
      nn.ReLU(),
      nn.Flatten(), # 16 * 16 * 32
      nn.Linear(16 * 16 * 32, 1000),
      nn.ReLU(),
      nn.Linear(1000, action_dim)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(3, 16, 7, 2, 3), # 32, 32
      nn.MaxPool2d(2), # 16, 16
      nn.ReLU(),
      nn.Conv2d(16, 64, 3, 2, 1), # 8, 8
      nn.MaxPool2d(2), # 4, 4
      nn.Flatten(), # 16
      nn.Linear(4 * 4 * 64, action_dim)
    )


def get_action(action_dim, state):
  if np.random.rand() < args.epsilon:
    action = np.random.randint(action_dim)
  else:
    model.eval()
    Q = model(state.unsqueeze(0).to(device))
    action = torch.max(Q, 1)[1].item()
  return action

def optimize():
  # train using past transitions
  model.train()
  optimizer.zero_grad()
  # randomly sample a batch of transitions from buffer
  batch = buffer.sample(args.batch_size, device)

  # predicted value of next state
  Qs = model(batch.state)
  outputs = Qs[np.arange(args.batch_size), batch.action]
  # estimated value of next state
  next_Qs = torch.max(model(batch.next_state), 1)[0]
  targets = args.gamma * batch.reward + next_Qs * ~batch.done

  # calculate loss
  loss = loss_func(outputs, targets)
  loss.backward()
  optimizer.step()

  # log loss
  #args.losses.append(loss)

class Args: pass
args = Args()
args.lr = 0.003
args.l2reg = 0.0003
args.episodes = 100
args.max_steps = 800
args.epsilon = 0.1
args.batch_size = 32
args.gamma = 0.99
args.buffer_size = 30000
args.losses = []
args.actions = []
args.rewards = []
args.reset = False

buffer = ExpBuffer(args.buffer_size)

env = gym.make('Smash-dk-v0')
action_dim = 15

model = get_DQN(action_dim)
if not args.reset:
    buffer = load_training(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
# Huber Loss: equivalent to MSE when difference is small, but less punishing
# when the difference is large, resilient to outliers
loss_func = F.smooth_l1_loss

# %%
def train():
    for episode_idx in range (1, args.episodes+1):
        state = env.reset()
        state = process_obs(state)
        buffer.add_state(state)
        score = 0

        for t in range(args.max_steps):
            # calculate next action using epsilon-greedy
            action = get_action(action_dim, state)

            # step
            state, reward, done, info = env.step(map_action_space(action))
            state = process_obs(state)
            buffer.add_record(action, state, reward, done)
            score += reward
            
            optimize()

            # log r, a
            #args.rewards.append(reward)
            #args.actions.append(action)

        if episode_idx % 10 == 0:
            save_training(model, buffer)
            save_stats(args)
            save_video(buffer, episode_idx)

        
        print('episode %d, score %f' % (episode_idx, score))

#train()
'''