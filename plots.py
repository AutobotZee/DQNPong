from DQN_v12 import *
import pickle
import matplotlib.pyplot as plt
file=['Dec1/Dec1cpu','Dec1/Dec1cpu_double','Dec1/Dec1cpu_duel',
      'Dec1/Dec1cpu_doubleduel','Dec2/Dec2cpu_cat1',
      'Dec2/Dec2cpu_doubleduel_huber',
      'Dec2/Dec2cpudddww','Dec2/Dec2cpu_double_cat1',
      'Dec2/Dec2cpu_double_cat2','Dec2/Dec2cpuv0','Dec2/Dec2cpuv0_double']
P=len(file)
policy=[True for i in range(P)];target=[True for i in range(P)];dur=[True for i in range(P)]
T=[True for i in range(P)];frames=[True for i in range(P)];R=[True for i in range(P)];
steps_done=[True for i in range(P)];
for i in range(P):
  with open(file[i]+'.pickle', 'rb') as f:
    policy[i],target[i],T[i],frames[i],dur[i],R[i],steps_done[i],_,_=pickle.load(f)
    
col=['r-','b-','g-','m-','k-','r-','b-','g-','m-']
dqname={0:'DQN',1:'Double DQN',2:'Dueling DQN',3:'Double Dueling',
        4:'Stack 1 DQN',5:'Huber Double Dueling',6:'MSE Double Dueling',
        7:'Stack 1 Double DQN',8:'Stack 2 Double DQN',9:'v0 DQN',10:'v0 Double'}


# plots
plt.figure()
for i in range(4):
  plt.plot(np.array(frames[i])/10**3,movave(R[i],15),col[i],label=dqname[i])
  plt.plot(np.array(frames[i])/10**3,R[i],col[i],alpha=0.2)
plt.xlabel('Frames [×1000]',fontsize=14);
plt.ylabel('Rewards',fontsize=14);
plt.tick_params(labelsize=14);
plt.xlim([0,1000]);
plt.legend(fontsize=11);
#plt.title('Double DQN, Dueling DQN',fontsize=15)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.savefig('Difmodelpdf.pdf');plt.show()

plt.figure()
for i in [1,7,8]:
  plt.plot(np.array(frames[i])/10**3,movave(R[i],15),col[i],label=dqname[i])
  plt.plot(np.array(frames[i])/10**3,R[i],col[i],alpha=0.2)
plt.xlabel('Frames [×1000]',fontsize=18);plt.ylabel('Rewards',fontsize=18);
plt.tick_params(labelsize=18);plt.xlim([0,1000]);plt.legend(fontsize=13);
#plt.title('Loss MSE vs Huber')
plt.savefig('loss_comp.pdf');plt.show()

plt.figure()
plt.plot(np.array(frames[7])/10**3,movave(R[7],15),col[7],label='1 frame')
plt.plot(np.array(frames[7])/10**3,R[7],col[7],alpha=0.2)
plt.plot(np.array(frames[8])/10**3,movave(R[8],15),col[8],label='2 frames')
plt.plot(np.array(frames[8])/10**3,R[8],col[8],alpha=0.2)
plt.plot(np.array(frames[1])/10**3,movave(R[1],15),col[1],label='4 frames')
plt.plot(np.array(frames[1])/10**3,R[1],col[1],alpha=0.2)
plt.xlabel('Frames [×1000]',fontsize=14);plt.ylabel('Rewards',fontsize=14);
plt.tick_params(labelsize=14);plt.xlim([0,1000]);plt.legend(fontsize=13);
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.title('Stack DQN')
plt.savefig('catpdf.pdf');plt.show()

# =============================================================================
# Durations
# =============================================================================

# plots
plt.figure(figsize=(6.4, 4.8))
for i in range(4):
  plt.plot(movave(np.array(dur[i])/10**3,15),col[i],label=dqname[i])
  plt.plot(np.array(dur[i])/10**3,col[i],alpha=0.2)
plt.xlabel('Episodes',fontsize=14);
plt.ylabel('Durations (Frames) [×1000]',fontsize=14);
plt.tick_params(labelsize=14);
plt.xlim([0,175]);
plt.legend(fontsize=11);
#plt.title('Episode durations',fontsize=20)
plt.savefig('dur.pdf');plt.show()

plt.figure()
for i in [1,7]:
  plt.plot(movave(dur[i],15),col[i],label=dqname[i])
  plt.plot(dur[i],col[i],alpha=0.2)
plt.xlabel('Episodes',fontsize=18);plt.ylabel('Durations (Frames)',fontsize=18);
plt.tick_params(labelsize=18);plt.xlim([0,1000]);plt.legend(fontsize=13);
#plt.title('Loss MSE vs Huber')
plt.savefig('loss_comp_dur.pdf');plt.show()

plt.figure()
plt.plot(movave(dur[1],15),col[1],label='4 frames input')
plt.plot(dur[1],col[1],alpha=0.2)
plt.plot(movave(dur[7],15),col[7],label='1 frame input')
plt.plot(dur[7],col[7],alpha=0.2)
plt.xlabel('Episodes',fontsize=18);plt.ylabel('Durations (Frames)',fontsize=18);
plt.tick_params(labelsize=18);plt.xlim([0,150]);plt.legend(fontsize=13);
#plt.title('Stack DQN')
plt.savefig('cat_dur.pdf');plt.show()

# =============================================================================
# =============================================================================
# steps epsilon
# =============================================================================
# =============================================================================

def eps(step):
  eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)
  return eps_threshold

from itertools import count
num_scr=4
steps=0
E=[]
for i in range(len(durations)):
  t=0
  for k in range(num_scr-1):
    t+=1
    E.append(eps(steps))
  for m in count():
    for n in range(num_scr):
      E.append(eps(steps))      
    steps+=1
    t+=1
    if t==durations[i]:
      break

plt.figure()
plt.plot(E)
plt.xscale('linear')
plt.tick_params(labelsize=18)
plt.xlim(0,len(E))
plt.xlabel('Steps',fontsize=18);plt.ylabel('Epsilon',fontsize=18);
plt.savefig('epsilon.png')
plt.show()



