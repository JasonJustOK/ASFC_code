from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn import functional as F
import time
import torch.nn as nn


import random
import numpy as np
from torch.autograd import Variable
import torch
from easydict import EasyDict as edict
from cfgs import model_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy_to_tensor(x):
    return Variable(torch.from_numpy(x)).float().to(device)

def generate_robot_action(robot_state, robot_actor_hidden, robot_critic_hidden, robot_network, inter_state):
    self_info = numpy_to_tensor(robot_state['self_info'])
    grid_map = numpy_to_tensor(robot_state['grid_map'])
    neighbor_info = numpy_to_tensor(robot_state['neighbor_info'])
    neighbor_mask = numpy_to_tensor(robot_state['neighbor_mask'])
    inter_info = numpy_to_tensor(robot_state['inter_info'])
    inter_mask = numpy_to_tensor(robot_state['inter_mask'])
    absolute_coordinate = numpy_to_tensor(robot_state['absolute_coordinate'])

    ii_self_info = numpy_to_tensor(inter_state['self_info'])
    ii_grid_map = numpy_to_tensor(inter_state['grid_map'])
    ii_neighbor_info = numpy_to_tensor(inter_state['neighbor_info'])
    ii_neighbor_mask = numpy_to_tensor(inter_state['neighbor_mask'])
    ii_robot_info = numpy_to_tensor(inter_state['robot_info'])
    ii_robot_mask = numpy_to_tensor(inter_state['robot_mask'])
    ii_absolute_coordinate = numpy_to_tensor(inter_state['absolute_coordinate'])


    robot_values, robot_actions, robot_action_log_probs, robot_actor_hidden, robot_critic_hidden = \
            robot_network(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_critic_hidden, absolute_coordinate)

    robot_values, robot_actions, robot_action_log_probs = robot_values.data.cpu().numpy(), robot_actions.data.cpu().numpy(), robot_action_log_probs.data.cpu().numpy()

    return robot_values, robot_actions, robot_action_log_probs, robot_actor_hidden, robot_critic_hidden


def generate_inter_action(inter_state, inter_critic_hidden, inter_network, robot_state):
    self_info = numpy_to_tensor(inter_state['self_info'])
    grid_map = numpy_to_tensor(inter_state['grid_map'])
    neighbor_info = numpy_to_tensor(inter_state['neighbor_info'])
    neighbor_mask = numpy_to_tensor(inter_state['neighbor_mask'])
    robot_info = numpy_to_tensor(inter_state['robot_info'])
    robot_mask = numpy_to_tensor(inter_state['robot_mask'])
    absolute_coordinate = numpy_to_tensor(inter_state['absolute_coordinate'])

    rr_self_info = numpy_to_tensor(robot_state['self_info']) 
    rr_grid_map = numpy_to_tensor(robot_state['grid_map'])
    rr_neighbor_info = numpy_to_tensor(robot_state['neighbor_info'])
    rr_neighbor_mask = numpy_to_tensor(robot_state['neighbor_mask'])
    rr_robot_info = numpy_to_tensor(robot_state['inter_info'])
    rr_robot_mask = numpy_to_tensor(robot_state['inter_mask'])
    rr_absolute_coordinate = numpy_to_tensor(robot_state['absolute_coordinate'])


    inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden = \
            inter_network(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden)

    inter_values, inter_actions, inter_action_log_probs = inter_values.data.cpu().numpy(), inter_actions.data.cpu().numpy(), inter_action_log_probs.data.cpu().numpy()

    return inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden



def add_one_buffer_robot(buffer, robot_state, inter_state, robot_actor_hidden, robot_critic_hidden, \
                            robot_values, robot_actions, robot_action_log_probs,\
                                robot_rewards, robot_dones, \
                                label_target, made_action):
    one_data = (
        robot_state,
        inter_state,
        robot_actor_hidden,
        robot_critic_hidden,
        robot_values,
        robot_actions,
        robot_action_log_probs,
        np.asarray(robot_rewards),
        np.asarray(robot_dones),

        label_target,
        made_action
        )

    buffer.append(one_data)

    return

def add_one_buffer_inter(buffer_inter, inter_state, robot_state, inter_critic_hidden, \
                            inter_values, inter_actions, inter_action_log_probs, \
                                inter_rewards, inter_dones):
    one_data = (
        inter_state, \
        robot_state, \
        inter_critic_hidden, \
        inter_values, \
        inter_actions, \
        inter_action_log_probs, \
        inter_rewards, \
        inter_dones
        )

    buffer_inter.append(one_data)

    return


def transform_robot_buffer_to_batch(buffer):

    robot_state_batch = []
    inter_state_batch = []
    robot_actor_hidden_batch = []
    robot_critic_hidden_batch = []
    robot_values_batch = []
    robot_actions_batch = []
    robot_action_log_probs_batch = []
    robot_rewards_batch = []
    robot_dones_batch = []

    label_target_batch = []
    made_action_batch = []


    robot_batch_data = {}

    for data in buffer:
        robot_state_batch.append(data[0])
        inter_state_batch.append(data[1])
        robot_actor_hidden_batch.append(data[2])
        robot_critic_hidden_batch.append(data[3])
        robot_values_batch.append(data[4])
        robot_actions_batch.append(data[5])
        robot_action_log_probs_batch.append(data[6])
        robot_rewards_batch.append(data[7])
        robot_dones_batch.append(data[8])

        label_target_batch.append(data[9])
        made_action_batch.append(data[10])

    robot_state_batch = np.asarray(robot_state_batch)
    inter_state_batch = np.asarray(inter_state_batch)
    robot_actor_hidden_batch = np.asarray(robot_actor_hidden_batch)
    robot_critic_hidden_batch = np.asarray(robot_critic_hidden_batch)

    robot_values_batch = np.asarray(robot_values_batch)
    robot_actions_batch = np.asarray(robot_actions_batch)
    robot_action_log_probs_batch = np.asarray(robot_action_log_probs_batch)
    robot_rewards_batch = np.asarray(robot_rewards_batch)
    robot_dones_batch = np.asarray(robot_dones_batch)

    robot_batch_data['robot_state_batch'] = robot_state_batch
    robot_batch_data['inter_state_batch'] = inter_state_batch
    robot_batch_data['robot_actor_hidden_batch'] = robot_actor_hidden_batch
    robot_batch_data['robot_critic_hidden_batch'] = robot_critic_hidden_batch
    robot_batch_data['robot_values_batch'] = robot_values_batch
    robot_batch_data['robot_actions_batch'] = robot_actions_batch
    robot_batch_data['robot_action_log_probs_batch'] = robot_action_log_probs_batch
    robot_batch_data['robot_rewards_batch'] = robot_rewards_batch
    robot_batch_data['robot_dones_batch'] = robot_dones_batch

    robot_batch_data['label_target_batch'] = np.asarray(label_target_batch)
    robot_batch_data['made_action_batch'] = np.asarray(made_action_batch)

    return robot_batch_data



def generate_robot_tar_adv_batch(robot_batch_data, last_robot_v):
    # calculate the target advantage
    rewards = robot_batch_data['robot_rewards_batch']
    values = robot_batch_data['robot_values_batch']
    dones = robot_batch_data['robot_dones_batch']

    num_step = rewards.shape[0]
    num_robots = rewards.shape[1]

    values = list(values)
    values.append(last_robot_v)
    values = np.asarray(values).reshape((num_step + 1, num_robots))

    target_batch = np.zeros((num_step, num_robots))
    gae = np.zeros((num_robots, ))

    gamma = model_cfg.GAMMA
    lam = model_cfg.LAMDA

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t+1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        target_batch[t, :] = gae + values[t, :]

    adv_batch = target_batch - values[:-1, :]

    return target_batch, adv_batch




class Env():
    # Note: This is a user-defined class, used for agent-env interaction. Please use your own env for training.
    def __init__(self):
        self.random_seed = model_cfg.seed
        self.room_random_state = np.random.RandomState(self.random_seed)

        random.seed(self.random_seed)

        self.num_robots = model_cfg.robot_num
        self.robots = [] 
        self.robots_ids = [] 

        self.obstacle_ids = []
        self.obs_object_ids = []

        self.num_inters = model_cfg.interferer_num
        self.inters = []
        self.inters_ids = []
        self.target = None
        self.target_id = None

        self.action_interval = model_cfg.action_interval

        self.timestep = 1

    def reset(self):
        # This func is for reset the env.
        self.timestep = 0
        return

    def get_dual_state(self):
        # This func is for get the states for robots and interferers, please use new simulator.

        robot_state = {}
        inter_state = {}

        robot_state['self_info'] = np.random.rand(self.num_robots, 4)
        robot_state['grid_map'] = np.random.rand(self.num_robots, 3, 33, 33)
        robot_state['neighbor_info'] = np.random.rand(self.num_robots, 10, 3)
        robot_state['neighbor_mask'] = np.random.rand(self.num_robots, 10)
        robot_state['inter_info'] = np.random.rand(self.num_robots, 10, 3)
        robot_state['inter_mask'] = np.random.rand(self.num_robots, 10)
        robot_state['absolute_coordinate'] = np.random.rand(self.num_robots, 2)


        inter_state['self_info'] = np.random.rand(self.num_inters, 2)
        inter_state['grid_map'] = np.random.rand(self.num_inters, 3, 33, 33)
        inter_state['neighbor_info'] = np.random.rand(self.num_inters, 10, 3)
        inter_state['neighbor_mask'] = np.random.rand(self.num_inters, 10)
        inter_state['robot_info'] = np.random.rand(self.num_inters, 10, 3)
        inter_state['robot_mask'] = np.random.rand(self.num_inters, 10)
        inter_state['absolute_coordinate'] = np.random.rand(self.num_inters, 2)


        label_target = np.random.rand(self.num_robots, 33, 33)

        return robot_state, inter_state, label_target


    def step(self, robot_actions, inter_actions):
        # execute the actions and receive the rewards.

        robot_rewards = [1.0 for i in range(self.num_robots)]
        inter_rewards = [1.0 for i in range(self.num_inters)]

        if self.timestep > 200:
            robot_dones = [True for i in range(self.num_robots)]
            inter_dones = [True for i in range(self.num_inters)]
        else:
            robot_dones = [False for i in range(self.num_robots)]
            inter_dones = [False for i in range(self.num_inters)]

        robot_infos, inter_infos = None, None

        self.timestep += 1

        return robot_rewards, robot_dones, robot_infos, inter_rewards, inter_dones, inter_infos

    def get_real_action(self, robot_actions):
        return np.random.rand(self.num_robots, 2)



def ppo_update_robot(global_update, robot_network, robot_batch_data):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    robot_state_batch = robot_batch_data['robot_state_batch']
    inter_state_batch = robot_batch_data['inter_state_batch']
    robot_actor_hidden_batch = robot_batch_data['robot_actor_hidden_batch']
    robot_critic_hidden_batch = robot_batch_data['robot_critic_hidden_batch']
    robot_values_batch = robot_batch_data['robot_values_batch']
    robot_actions_batch = robot_batch_data['robot_actions_batch']
    robot_action_log_probs_batch = robot_batch_data['robot_action_log_probs_batch']
    robot_rewards_batch = robot_batch_data['robot_rewards_batch']
    robot_dones_batch = robot_batch_data['robot_dones_batch']

    t_batch = robot_batch_data['t_batch']
    adv_batch = robot_batch_data['adv_batch']

    label_target_batch = robot_batch_data['label_target_batch']
    made_action_batch = robot_batch_data['made_action_batch']

    epoch = model_cfg.EPOCH
    batch_size = model_cfg.BATCH_SIZE
    clip_param = model_cfg.clip_param
    value_loss_coef = model_cfg.value_loss_coef
    action_loss_coef = model_cfg.action_loss_coef
    entropy_coef = model_cfg.entropy_coef
    aux_loss_coef = model_cfg.aux_loss_coef
    max_grad_norm = model_cfg.max_grad_norm


    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-5)

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0
    aux_loss_epoch = 0

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(adv_batch.shape[0]))), batch_size=batch_size, drop_last=False)
        for i, index in enumerate(sampler):
            sampled_robot_state_batch = robot_state_batch[index]
            sampled_inter_state_batch = inter_state_batch[index]

            sampled_robot_actor_hidden_batch = robot_actor_hidden_batch[index]
            sampled_robot_critic_hidden_batch = robot_critic_hidden_batch[index]

            sampled_robot_actions_batch = robot_actions_batch[index]

            sampled_made_action_batch = made_action_batch[index]

            values, action_log_probs, dist_entropy, pred_logits_map = evaluate_actions_helper(robot_network, \
                                                                            sampled_robot_state_batch, sampled_inter_state_batch, \
                                                                            sampled_robot_actor_hidden_batch, sampled_robot_critic_hidden_batch, \
                                                                            sampled_robot_actions_batch, \
                                                                            sampled_made_action_batch)


            pred_logits_map = torch.flatten(pred_logits_map, start_dim=0, end_dim=1)
            label_target = Variable(torch.from_numpy(label_target_batch[index])).float().to(device)
            label_target = torch.flatten(label_target, start_dim=0, end_dim=1) 

            # weight:
            label_target_weight = torch.tensor([1., 5., 10., 15.]).to(device)
            # print(label_target_weight)
            aux_loss = cross_entropy2d(pred_logits_map, label_target.long(), label_target_weight) 


            values = values.view(-1, 1) 
            action_log_probs = action_log_probs.view(-1, 1)
            dist_entropy = dist_entropy.view(-1, 1)


            value_preds = Variable(torch.from_numpy(robot_values_batch[index])).float().to(device).view(-1, 1)
            old_action_log_probs = Variable(torch.from_numpy(robot_action_log_probs_batch[index])).float().to(device).view(-1, 1)
            returns = Variable(torch.from_numpy(t_batch[index])).float().to(device).view(-1, 1)
            adv_targ = Variable(torch.from_numpy(adv_batch[index])).float().to(device).view(-1, 1)


            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * adv_targ


            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ


            action_loss = -torch.min(surr1, surr2).mean()


            action_clip = (torch.abs(ratio - 1.0) <= clip_param).float().mean().item() 


            value_pred_clipped = value_preds + \
                                (values - value_preds).clamp(
                                    -clip_param, clip_param)


            value_losses = (values - returns).pow(2)



            value_losses_clipped = (value_pred_clipped
                                    - returns).pow(2)


            value_loss = .5 * torch.max(value_losses,
                                        value_losses_clipped).mean()


            value_clip = (torch.abs(values - value_preds) <= clip_param).float().mean().item()



            robot_network.actor_optimizer.zero_grad()
            robot_network.critic_optimizer.zero_grad()


            (value_loss * value_loss_coef + action_loss * action_loss_coef - dist_entropy * entropy_coef + aux_loss * aux_loss_coef).backward()


            nn.utils.clip_grad_norm_(robot_network.actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(robot_network.critic.parameters(), max_grad_norm)

            robot_network.actor_optimizer.step()
            robot_network.critic_optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()
            aux_loss_epoch += aux_loss.item()

    num_updates = epoch * batch_size

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates
    aux_loss_epoch /= num_updates


    print('~~'.join(map(str, [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
        'trainning:', 'epoch * batch_size_global_update + action_loss_epoch + value_loss_epoch + dist_entropy_epoch + aux_loss_epoch:', \
        global_update, action_loss_epoch, value_loss_epoch, dist_entropy_epoch, aux_loss_epoch])))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch



def evaluate_actions_helper(robot_network, \
                                sampled_robot_state_batch, sampled_inter_state_batch, \
                                sampled_robot_actor_hidden_batch, sampled_robot_critic_hidden_batch, \
                                sampled_robot_actions_batch, \
                                sampled_made_action_batch):
    
    all_values = []
    all_action_log_probs = []
    all_dist_entropy = []
    all_pred_logits_map = []

    for i in range(sampled_robot_state_batch.shape[0]):
        robot_state = sampled_robot_state_batch[i]
        self_info = numpy_to_tensor(robot_state['self_info'])
        grid_map = numpy_to_tensor(robot_state['grid_map'])
        neighbor_info = numpy_to_tensor(robot_state['neighbor_info'])
        neighbor_mask = numpy_to_tensor(robot_state['neighbor_mask'])
        inter_info = numpy_to_tensor(robot_state['inter_info'])
        inter_mask = numpy_to_tensor(robot_state['inter_mask'])
        absolute_coordinate = numpy_to_tensor(robot_state['absolute_coordinate'])


        robot_actor_hidden = sampled_robot_actor_hidden_batch[i]
        robot_critic_hidden = sampled_robot_critic_hidden_batch[i]

        robot_actions = numpy_to_tensor(sampled_robot_actions_batch[i])

        made_action = numpy_to_tensor(sampled_made_action_batch[i])

        values, action_log_probs, dist_entropy, pred_logits_map = \
                robot_network.evaluate_actions(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_actor_hidden, absolute_coordinate, \
                    robot_actions, \
                    made_action)


        all_values.append(values)
        all_action_log_probs.append(action_log_probs)
        all_dist_entropy.append(dist_entropy)
        all_pred_logits_map.append(pred_logits_map)

    all_values = torch.stack(all_values) 
    all_action_log_probs = torch.stack(all_action_log_probs) 
    all_dist_entropy = torch.stack(all_dist_entropy)
    all_dist_entropy = all_dist_entropy.mean() # 1
    all_pred_logits_map = torch.stack(all_pred_logits_map)

    return all_values, all_action_log_probs, all_dist_entropy, all_pred_logits_map


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density


def cross_entropy2d(input, target, weight=None, size_average=True):

    n, c, h, w = input.size()

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0

    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()

    return loss


def transform_inter_buffer_to_batch(buffer_inter):

    inter_state_batch = []
    robot_state_batch = []
    inter_critic_hidden_batch = []
    inter_values_batch = []
    inter_actions_batch = []
    inter_action_log_probs_batch = []
    inter_rewards_batch = []
    inter_dones_batch = []

    inter_batch_data = {}

    for data in buffer_inter:
        inter_state_batch.append(data[0])
        robot_state_batch.append(data[1])
        inter_critic_hidden_batch.append(data[2])
        inter_values_batch.append(data[3])
        inter_actions_batch.append(data[4])
        inter_action_log_probs_batch.append(data[5])
        inter_rewards_batch.append(data[6])
        inter_dones_batch.append(data[7])

    inter_state_batch = np.asarray(inter_state_batch)
    robot_state_batch = np.asarray(robot_state_batch)
    inter_critic_hidden_batch = np.asarray(inter_critic_hidden_batch)

    inter_values_batch = np.asarray(inter_values_batch)
    inter_actions_batch = np.asarray(inter_actions_batch)
    inter_action_log_probs_batch = np.asarray(inter_action_log_probs_batch)
    inter_rewards_batch = np.asarray(inter_rewards_batch)
    inter_dones_batch = np.asarray(inter_dones_batch)

    inter_batch_data['inter_state_batch'] = inter_state_batch
    inter_batch_data['robot_state_batch'] = robot_state_batch
    inter_batch_data['inter_critic_hidden_batch'] = inter_critic_hidden_batch
    inter_batch_data['inter_values_batch'] = inter_values_batch
    inter_batch_data['inter_actions_batch'] = inter_actions_batch
    inter_batch_data['inter_action_log_probs_batch'] = inter_action_log_probs_batch
    inter_batch_data['inter_rewards_batch'] = inter_rewards_batch
    inter_batch_data['inter_dones_batch'] = inter_dones_batch

    return inter_batch_data

def generate_inter_tar_adv_batch(inter_batch_data, last_inter_v):
    rewards = inter_batch_data['inter_rewards_batch']
    values = inter_batch_data['inter_values_batch']
    dones = inter_batch_data['inter_dones_batch']

    num_step = rewards.shape[0]
    num_inters = rewards.shape[1]

    values = list(values)
    values.append(last_inter_v)
    values = np.asarray(values).reshape((num_step + 1, num_inters)) 

    target_batch = np.zeros((num_step, num_inters))
    gae = np.zeros((num_inters, ))

    gamma = model_cfg.GAMMA 
    lam = model_cfg.LAMDA

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t+1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        target_batch[t, :] = gae + values[t, :]

    adv_batch = target_batch - values[:-1, :]

    return target_batch, adv_batch


def generate_inter_action_sample(inter_state, inter_critic_hidden, inter_network, robot_state, sampled_policy):
    self_info = numpy_to_tensor(inter_state['self_info'])
    grid_map = numpy_to_tensor(inter_state['grid_map'])
    neighbor_info = numpy_to_tensor(inter_state['neighbor_info'])
    neighbor_mask = numpy_to_tensor(inter_state['neighbor_mask'])
    robot_info = numpy_to_tensor(inter_state['robot_info'])
    robot_mask = numpy_to_tensor(inter_state['robot_mask'])
    absolute_coordinate = numpy_to_tensor(inter_state['absolute_coordinate'])

    rr_self_info = numpy_to_tensor(robot_state['self_info']) 
    rr_grid_map = numpy_to_tensor(robot_state['grid_map'])
    rr_neighbor_info = numpy_to_tensor(robot_state['neighbor_info'])
    rr_neighbor_mask = numpy_to_tensor(robot_state['neighbor_mask'])
    rr_robot_info = numpy_to_tensor(robot_state['inter_info'])
    rr_robot_mask = numpy_to_tensor(robot_state['inter_mask'])
    rr_absolute_coordinate = numpy_to_tensor(robot_state['absolute_coordinate'])


    inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden = \
            inter_network(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden)

    inter_values, inter_actions, inter_action_log_probs = inter_values.data.cpu().numpy(), inter_actions.data.cpu().numpy(), inter_action_log_probs.data.cpu().numpy()

    return inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden


def ppo_update_inter(global_update_inter, inter_network, inter_batch_data):
    assert len(inter_batch_data) == 8 + 2
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    inter_state_batch = inter_batch_data['inter_state_batch']
    robot_state_batch = inter_batch_data['robot_state_batch']
    inter_critic_hidden_batch = inter_batch_data['inter_critic_hidden_batch']
    inter_values_batch = inter_batch_data['inter_values_batch']
    inter_actions_batch = inter_batch_data['inter_actions_batch']
    inter_action_log_probs_batch = inter_batch_data['inter_action_log_probs_batch']
    inter_rewards_batch = inter_batch_data['inter_rewards_batch']
    inter_dones_batch = inter_batch_data['inter_dones_batch']


    t_batch = inter_batch_data['t_batch']
    adv_batch = inter_batch_data['adv_batch']

    epoch = model_cfg.EPOCH
    batch_size = model_cfg.BATCH_SIZE
    clip_param = model_cfg.clip_param
    value_loss_coef = model_cfg.value_loss_coef
    action_loss_coef = model_cfg.action_loss_coef
    entropy_coef = model_cfg.entropy_coef
    max_grad_norm = model_cfg.max_grad_norm


    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-5)

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(adv_batch.shape[0]))), batch_size=batch_size, drop_last=False)
        for i, index in enumerate(sampler):
            sampled_inter_state_batch = inter_state_batch[index] 
            sampled_robot_state_batch = robot_state_batch[index] 

            sampled_inter_critic_hidden_batch = inter_critic_hidden_batch[index]

            sampled_inter_actions_batch = inter_actions_batch[index]

            values, action_log_probs, dist_entropy = evaluate_actions_inter_helper(inter_network, \
                                                                            sampled_inter_state_batch, sampled_robot_state_batch, \
                                                                            sampled_inter_critic_hidden_batch, \
                                                                            sampled_inter_actions_batch)

            values = values.view(-1, 1)
            action_log_probs = action_log_probs.view(-1, 1)
            dist_entropy = dist_entropy.view(-1, 1)


            value_preds = Variable(torch.from_numpy(inter_values_batch[index])).float().to(device).view(-1, 1)
            old_action_log_probs = Variable(torch.from_numpy(inter_action_log_probs_batch[index])).float().to(device).view(-1, 1)
            returns = Variable(torch.from_numpy(t_batch[index])).float().to(device).view(-1, 1)
            adv_targ = Variable(torch.from_numpy(adv_batch[index])).float().to(device).view(-1, 1)


            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * adv_targ


            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ


            action_loss = -torch.min(surr1, surr2).mean()


            action_clip = (torch.abs(ratio - 1.0) <= clip_param).float().mean().item()  # no use


            # value
            value_pred_clipped = value_preds + \
                                (values - value_preds).clamp(
                                    -clip_param, clip_param)


            value_losses = (values - returns).pow(2)



            value_losses_clipped = (value_pred_clipped
                                    - returns).pow(2)


            value_loss = .5 * torch.max(value_losses,
                                        value_losses_clipped).mean()
            value_clip = (torch.abs(values - value_preds) <= clip_param).float().mean().item() 



            inter_network.optimizer.zero_grad()

            (value_loss * value_loss_coef + action_loss * action_loss_coef - dist_entropy * entropy_coef).backward()

            nn.utils.clip_grad_norm_(inter_network.actor_and_critic.parameters(), max_grad_norm)

            inter_network.optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

    num_updates = epoch * batch_size

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates


    print('~~'.join(map(str, [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
        'inter_trainning:', 'epoch * batch_size_global_update + action_loss_epoch + value_loss_epoch + dist_entropy_epoch:', global_update_inter, \
        action_loss_epoch, value_loss_epoch, dist_entropy_epoch])))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch



def evaluate_actions_inter_helper(inter_network, \
                                    sampled_inter_state_batch, sampled_robot_state_batch, \
                                    sampled_inter_critic_hidden_batch, \
                                    sampled_inter_actions_batch):
    all_values = []
    all_action_log_probs = []
    all_dist_entropy = []

    for i in range(sampled_inter_state_batch.shape[0]):
        inter_state = sampled_inter_state_batch[i]
        # robot_state = sampled_robot_state_batch[i]
        self_info = numpy_to_tensor(inter_state['self_info'])
        grid_map = numpy_to_tensor(inter_state['grid_map'])
        neighbor_info = numpy_to_tensor(inter_state['neighbor_info'])
        neighbor_mask = numpy_to_tensor(inter_state['neighbor_mask'])
        robot_info = numpy_to_tensor(inter_state['robot_info'])
        robot_mask = numpy_to_tensor(inter_state['robot_mask'])
        absolute_coordinate = numpy_to_tensor(inter_state['absolute_coordinate'])


        inter_critic_hidden = sampled_inter_critic_hidden_batch[i]

        inter_actions = numpy_to_tensor(sampled_inter_actions_batch[i])

        values, action_log_probs, dist_entropy = \
                inter_network.evaluate_actions(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden, \
                                    inter_actions)



        all_values.append(values)
        all_action_log_probs.append(action_log_probs)
        all_dist_entropy.append(dist_entropy)

    all_values = torch.stack(all_values) 
    all_action_log_probs = torch.stack(all_action_log_probs) 
    all_dist_entropy = torch.stack(all_dist_entropy)
    all_dist_entropy = all_dist_entropy.mean() # 1

    return all_values, all_action_log_probs, all_dist_entropy




import os
import random

def traverse_folder(folder_path):
    stack = [folder_path]
    res = []
    while len(stack) > 0:
        path = stack.pop()
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                res.append(file_path)
            else:
                stack.append(file_path)

    return res




def random_pick(some_list,probabilities):
    x = random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item


def get_init_reward(policy_number, init):
    return [init for i in range(policy_number)]



def get_avg_reward(each_reward_i):
    l = len(each_reward_i)
    epi_r = 0
    for epi in each_reward_i:
        epi_r += sum(epi)

    return epi_r/l



def get_sampled_inter_policy(ALL_PATH, ALL_REWARD, SAMPLE_NUM):
    rew_sum = sum(ALL_REWARD)

    all_prob = [r/rew_sum for r in ALL_REWARD]
    res = []
    for i in range(SAMPLE_NUM):
        this_path = random_pick(ALL_PATH, all_prob)
        res.append(this_path)

    return res


