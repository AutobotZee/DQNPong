#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from gym import wrappers
import math
import random
import matplotlib
from collections import namedtuple
from itertools import count
from PIL import Image
import torchvision.transforms as T

env = gym.make('Pong-v0')
s= env.reset()
a = env.action_space.sample()

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x
def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


def show_replay():
    """
    Not-so-elegant way to display the MP4 file generated by the Monitor wrapper inside a notebook.
    The Monitor wrapper dumps the replay to a local file that we then display as a HTML video object.
    """
    import io
    import base64
    from IPython.display import HTML
    video = io.open('./gym-results/openaigym.video.%s.video000000.mp4' % env.file_infix, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''
        <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
    .format(encoded.decode('ascii')))
    
env = gym.make('Pong-v0')
env = wrappers.Monitor(env, "./gym-results", force=True) # Create wrapper to display environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run environment
env.reset() # Reset environment
# =============================================================================
# Reward=[];Obs=[];Done=[];Acs=[]
# while True:
#     env.render() # Render environment
#     action = env.action_space.sample() # Get a random action
#     observation, reward, done, info = env.step(action) # Take a step
#     Reward.append(reward)
#     Obs.append(observation)
#     Done.append(done)
#     Acs.append(action)
#     if done: break # Break if environment is done
# env.close() # Close environment
# =============================================================================
#show_replay()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
      
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:,34:194,:]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)
  

env.reset()

# =============================================================================
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
# plt.title('Example extracted screen')
# plt.show()
# 
# =============================================================================
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions)
target_net = DQN(screen_height, screen_width, n_actions)
if use_cuda:
    policy_net.cuda()
    target_net.cuda()
target_net.load_state_dict(policy_net.state_dict())

    
#target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
#optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return get_variable(policy_net(state).max(1)[1].view(1, 1))
    else:
        return get_variable(torch.tensor([[random.randrange(n_actions)]],  dtype=torch.long))

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),  dtype=torch.bool)
    non_final_mask = get_variable(non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = get_variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...');plt.xlabel('Episode');plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)# clear what is written on figure
        display.display(plt.gcf())# display 
        
num_episodes = 100
episode_durations = []


import pickle

## Dump the data into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump(num_episodes, f)

## Load the data from a pickle file
with open('data.pickle', 'rb') as f:
    num_episodes = pickle.load(f)


for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and state
    env.reset()
    
    s1 = get_screen()
    a1 = get_variable(torch.tensor([[random.randrange(n_actions)]], dtype=torch.long))
    _,_,d1,_=env.step(a1.item())
    s2 = get_screen()
    a2 = get_variable(torch.tensor([[random.randrange(n_actions)]], dtype=torch.long))
    _,_,d2,_=env.step(a1.item())
    s3 = get_screen()
    state = torch.cat((s1.squeeze(),s2.squeeze(),s3.squeeze())).unsqueeze(0) # tensor as a state
    
    if d1+d2 == 0:
        for t in count():
            if t % 100 == 0: print(t)
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            
            r = get_variable(torch.tensor([reward]))
    
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = get_variable(current_screen - last_screen)
            else:
                next_state = None
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
    #            plot_durations(episode_durations)
                break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')


env = wrappers.Monitor(env, "./gym-results", force=True) # Create wrapper to display environment
env.reset()
last_screen = get_screen()# tensor as a state
current_screen = get_screen()
state = get_variable(current_screen - last_screen) # the initial speed is zero

R=0
for _ in range(10000):
    env.render()
    action = policy_net(state).max(1)[1].view(1, 1)
    _, reward, done, _ = env.step(action.item())
    last_screen = current_screen
    current_screen = get_screen()
    state = get_variable(current_screen - last_screen)
    R = R+reward
    if done: break
  
env.close()
show_replay()









