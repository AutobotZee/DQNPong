# %matplotlib inline
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
import pickle
import cv2
from torch.autograd import Variable
import seaborn as sns

Duel='on'
Double='on'
game='PongNoFrameskip-v4'

env = gym.make(game)
num_scr=4


# Define the DQN model name
if (Double=='on') and (Duel=='on'):
    DQNname='Double Duel DQN'
elif (Double=='off') and (Duel=='on'):
    DQNname='Duel DQN'
elif (Double=='on') and (Duel=='off'):
    DQNname='Double DQN'
else:
    DQNname='normal DQN'

# Fixing random numbers
np.random.seed(42)
torch.manual_seed(42)


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

# Run environment
env.reset()# Reset environment

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, n, h, w, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU() )

        conv_out_size = self._get_conv_out(n,h,w)
        
        if Duel=='off':
            # Normal DQN
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)  )
        else:
            # Duel network
            self.value = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)  )
            self.advantage = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions) )

    def _get_conv_out(self, n,h,w):
        o = self.conv(Variable(torch.zeros(1,n,h,w)))# (1,*[4,84,84])=(1,4,84,84)
        return int(np.prod(o.size()))# product of all elements

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        # Normal DQN
        if Duel=='off':
            output = self.fc(conv_out)
        # Duel network
        else:            
            adv = self.advantage(conv_out)
            val = self.value(conv_out).expand(-1,adv.size(1))
            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

resize = T.Compose([T.ToPILImage(), T.Resize(84, interpolation=Image.CUBIC), T.ToTensor()])


def orig():
    screen = env.render(mode='rgb_array')
#    screen =  cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#    screen = screen[34:194,:]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).unsqueeze(0)
    return screen.unsqueeze(0)

def crop():
    screen = env.render(mode='rgb_array')
#    screen =  cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[34:194,:]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).unsqueeze(0)
    return screen.unsqueeze(0)

def noresize():
    screen = env.render(mode='rgb_array')
    screen =  cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[34:194,:]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).unsqueeze(0)
    return screen.unsqueeze(0)

def get_screen():
    screen = env.render(mode='rgb_array')
    screen =  cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[34:194,:]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).unsqueeze(0)
    return resize(screen).unsqueeze(0)




STEPS = 1
BATCH_SIZE = 32*STEPS
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 2*10**4
MEMORY_BOUND = 10000
TARGET_UPDATE = 1000

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(num_scr ,screen_height, screen_width, n_actions)
target_net = DQN(num_scr, screen_height, screen_width, n_actions)
## Loding previopus data of policy_net

if use_cuda:
    policy_net.cuda()
    target_net.cuda()
target_net.load_state_dict(policy_net.state_dict())

# target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(),lr=0.0001)
#optimizer = optim.Adam(policy_net.parameters(),lr=0.0001)
memory = ReplayMemory(100000)

def optimize_model():
    if len(memory) < MEMORY_BOUND:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = get_variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    if Double=='off':
        # Normal DQN
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    else:
        # Double DQN
        ad = policy_net(non_final_next_states).max(1)[1].view(-1,1).detach()
        qbar=get_variable(torch.zeros(BATCH_SIZE))
        qbar[non_final_mask]=target_net(non_final_next_states).gather(1, ad).view(-1).detach()
        expected_state_action_values = ( qbar*GAMMA ) + reward_batch
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()

def movave(r,n=10):
    X=[i for i in range(len(r))]
    for i in range(len(r)):
        if i+1<n:
            X[i]=sum(r[:i+1])/(i+1)
        else:
            X[i]=sum(r[i+1-n:i+1])/n
    return X


durations = []
R=[]
frames=[]
T=0
steps_done=0

def train(num_episodes):
    global R
    global durations
    global T
    global frames
    global steps_done
    global eps_threshold
    for i_episode in range(num_episodes):
        print('episode',len(durations))
        # Initialize the environment and state
        env.reset()
        s = [i for i in range(num_scr)]
        a = [i for i in range(num_scr-1)]
        d = [False for i in range(num_scr-1)]
        for i in range(num_scr-1):
            s[i]=get_screen()
            a[i]=get_variable(torch.tensor([[random.randrange(n_actions)]], dtype=torch.long))
            _,_,d[i],_=env.step(a[i].item())
        s[num_scr-1]=get_screen()
        state =get_variable( Variable(torch.cat([s[i].squeeze(0) for i in range(num_scr)]).unsqueeze(0)))
        Reward=0.0

        T=T+num_scr
        if sum(d)!=0:
            durations.append(0)
            R.append(Reward)
            frames.append(T)
        else:
            for t in count():
                # Select and perform an action
                #action = select_action(state)
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                if t % STEPS ==0:
                    steps_done +=1
                if sample > eps_threshold:
                    with torch.no_grad():
                        action = get_variable(policy_net(state).max(1)[1].view(1, 1))
                else:
                    action = get_variable(torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)) 
                # states update
                reward_cpu = 0.0
                ss=[i for i in range(num_scr)]
                dd=[False for i in range(num_scr)]
                for i in range(num_scr):
                    _, re, dd[i], _ =env.step(action.item())
                    T+=1
                    reward_cpu = reward_cpu + re 
                    ss[i]=get_screen()
                    if dd[i]: break
                # frames update
                reward = get_variable(torch.tensor([reward_cpu]))
                if t % 100 == 0:
                    print('t=',t,'/ Frames',T,'/ steps_done',steps_done,'/ eps',eps_threshold)
                # Observe new state
                if sum(dd)==0:
                    next_state = get_variable(torch.cat([ss[i].squeeze(0) for i in range(num_scr)]).unsqueeze(0))
                else:
                    next_state = None
            # Store the transition in memory
                memory.push(state, action, next_state, reward)
            # Move to the next state
                state = Variable(next_state)
                Reward = Reward+reward_cpu
            # Perform one step of the optimization (on the target network)
                if t % STEPS == 0:
                    optimize_model()
                if sum(dd)!=0:
                    durations.append(num_scr*(t+1)+num_scr)
                    R.append(Reward)
                    frames.append(T)
                    break
                if T % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
#              Update the target network, copying all weights and biases in DQN
#            if (len(durations)-1) % TARGET_UPDATE == 0:
#                target_net.load_state_dict(policy_net.state_dict())
        plt.figure()
        plt.plot(frames,R)
        plt.plot(frames,movave(R))
        plt.xlabel('Frames')
        plt.ylabel('Rewards')
        plt.title(DQNname)
        plt.show()
              
    print('Complete')
    # save data
    with open('models.pickle','wb') as f:
        pickle.dump((policy_net.state_dict(),target_net.state_dict(),T,frames,durations,R,steps_done,difficulty,STEPS),f)

def play(env):
    env = wrappers.Monitor(env, "./gym-results", force=True)  # Create wrapper to display environment
    env.reset()
    s = [i for i in range(num_scr)]
    a = [i for i in range(num_scr-1)]
    d = [False for i in range(num_scr-1)]
    for i in range(num_scr-1):
        s[i]=get_screen()
        a[i]=torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
        _,_,d[i],_=env.step(a[i].item())
    s[num_scr-1]=get_screen()
    state = torch.cat([s[i].squeeze(0) for i in range(num_scr)]).unsqueeze(0)
    state = get_variable(state)
    if sum(d)!=0:
        env.close()
        print('Initial state failed')
        return
    else:
        R = 0
        for _ in range(10000):
            env.render()
            action = policy_net(state).max(1)[1].view(1, 1)
            action = get_variable(action)
            ss=[i for i in range(num_scr)]
            dd=[False for i in range(num_scr)]
            for i in range(num_scr):
                _, reward, dd[i], _ = env.step(action.item())
                env.render()
                R = R+reward
                ss[i]=get_screen()
            state = torch.cat([ss[i].squeeze(0) for i in range(num_scr)]).unsqueeze(0)
            state = get_variable(state)
            if sum(dd)!=0:
                break
        env.close()
def keep_train(n):
    for i in range(1,n):
        train(3)

if game=='PongNoFrameskip-v4':
    difficulty='easy'
elif game=='PongNoFrameskip-v0':
    difficulty='difficult'
else:
    difficulty='other game'


