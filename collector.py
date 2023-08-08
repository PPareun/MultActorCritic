import numpy as np
import sys
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import torch
from actor_critic import ActorCritic
from torch.distributions.categorical import Categorical
import cv2

episode_id = np.random.randint(100000000)
worker_id = int(sys.argv[1])
# worker_id = 1

env = UnityEnvironment(file_name="C:\\Users\\pc\\Downloads\\Window_build", seed=episode_id, side_channels=[], worker_id=worker_id)
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]

model = ActorCritic()
state_dict = torch.load('alg.pt')
model.load_state_dict(state_dict)
model.cuda()
model.reset(1)

obs_agent0 = []
obs_agent1 = []

action_agent0 = []
action_agent1 = []

rewards = []
terminated = []

cnt = -1
print(f"start {worker_id}")


while True:
    cnt += 1
    decision_steps = env.get_steps(behavior_name)

    if len(decision_steps[1].reward):
        terminate = True
        ds = decision_steps[1]
    else:
        terminate = False
        ds = decision_steps[0]

    obs0 = ds.obs[0][0]
    obs1 = ds.obs[0][1]
    reward = ds.reward
    assert len(reward) == 2

    if cnt < 10 or reward[0] != reward[1]:
        reward = -1e-4
        rewards.append(reward)
    else:
        reward = reward[0]
        rewards.append(reward)

    obs_agent0.append(obs0)
    obs_agent1.append(obs1)

    if terminate is True:
        terminated.append(1)
        env.close()
        break
    else:
        terminated.append(0)

    with torch.no_grad():
        obs_temp = cv2.resize(obs0, dsize=(64, 64))
        obs_temp = torch.FloatTensor(obs_temp).unsqueeze(0).permute(0, 3, 1, 2)
        action0 = model(obs_temp.cuda()).logits_actions
        obs_temp1 = cv2.resize(obs1, dsize=(64, 64))
        obs_temp1 = torch.FloatTensor(obs_temp1).unsqueeze(0).permute(0, 3, 1, 2)
        action1 = model(obs_temp1.cuda()).logits_actions
        
    action0 =  Categorical(logits=action0).sample().item()
    action1 =  Categorical(logits=action1).sample().item()
    
    action_agent0.append(action0)
    action_agent1.append(action1)

    action_tuple0 = ActionTuple(np.array([[]], dtype=np.float32), np.array([[action0]], dtype=np.int32))
    action_tuple1 = ActionTuple(np.array([[]], dtype=np.float32), np.array([[action1]], dtype=np.int32))

    env.set_action_for_agent(behavior_name, 0, action_tuple0)
    env.set_action_for_agent(behavior_name, 1, action_tuple1)

    env.step()

obs_agent0 = (np.array(obs_agent0) * 255.).astype(np.uint8)
obs_agent1 = (np.array(obs_agent1) * 255.).astype(np.uint8)
np.savez_compressed('sheep/{}_obs0'.format(episode_id), np.array(obs_agent0))
np.savez_compressed('sheep/{}_obs1'.format(episode_id), np.array(obs_agent1))
np.savez_compressed('sheep/{}_action0'.format(episode_id), np.array(action_agent0))
np.savez_compressed('sheep/{}_action1'.format(episode_id), np.array(action_agent1))
np.savez_compressed('sheep/{}_rewards'.format(episode_id), np.array(rewards))

print(f"end {worker_id} {episode_id}")