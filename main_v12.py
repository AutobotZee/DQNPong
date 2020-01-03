from DQN_v12 import *

# load the saved data
# unless you clear all the varibles, you can ignore this and 
# you can just run train()
# change the name of .pickle file
with open('Dec1/Dec1cpu_doubleduel.pickle', 'rb') as f:
    policy,target,T,frames,durations,R,steps_done,difficulty,STEPS=pickle.load(f)
policy_net.load_state_dict(policy)
target_net.load_state_dict(target)

# train model
#train(4)

# play game
play(env)

# plots
plt.figure()
plt.plot(frames,movave(R,15),label='Moving Average')
plt.plot(frames,R,label='True Rewards',alpha=0.1)
plt.xlabel('Frames');plt.ylabel('Rewards')
plt.legend()
plt.title(DQNname)
plt.show()

for i in range(100):
  env.step(random.randrange(n_actions))
plt.figure()
plt.imshow(orig().cpu().squeeze().numpy(), interpolation='none')
plt.axis('off');plt.savefig('orig.pdf');plt.show()
plt.figure()
plt.imshow(crop().cpu().squeeze().numpy(), interpolation='none')
plt.axis('off');plt.savefig('crop.pdf');plt.show()
plt.figure()
plt.imshow(noresize().cpu().squeeze().numpy(), cmap='gist_gray',interpolation='none')
plt.axis('off');plt.savefig('noresize.pdf');plt.show()
plt.figure()
plt.imshow(get_screen().cpu().squeeze().numpy(), cmap='gist_gray',interpolation='none')
plt.axis('off');plt.savefig('resize.pdf');plt.show()















