import math
import numpy as np
import torch
from torch.optim import Adam
import time
import json
import sys

from utils import *

from ASFC import Robot_Network
from inters_network import Inter_Network


from cfgs import model_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path_prefix = sys.argv[1]

ALL_PATH = traverse_folder('./inters_model')
ALL_PATH = [str(i) for i in range(50)]

policy_number = len(ALL_PATH)

ALL_REWARD = get_init_reward(policy_number, model_cfg.sample_init_reward)

SAMPLE_NUM = model_cfg.interferer_num

env = Env()
robot_network = Robot_Network().to(device)


MAX_ROUND = 200
MAX_EPI = 30


epi_num = 0

for rd in range(MAX_ROUND):
    round_reward = {}
    round_reward['each_reward'] = [[] for i in range(SAMPLE_NUM)]
    round_reward['avg_reward'] = [0 for i in range(SAMPLE_NUM)]

    sampled_path = get_sampled_inter_policy(ALL_PATH, ALL_REWARD, SAMPLE_NUM)

    sampled_policy = []
    for p in sampled_path:
        inter_network = Inter_Network().to(device)

    print('New_sample_complete : ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    env.reset()
    robot_state, inter_state, _ = env.get_dual_state()

    robot_actor_hidden = torch.zeros(env.num_robots, 32).to(device)
    robot_critic_hidden = torch.zeros(env.num_robots, 32).to(device)

    inter_critic_hidden = torch.zeros(env.num_inters, 32).to(device)

    global_update, SMALL_EPI, buffer, global_update_inter, buffer_inter = 0, 0, [], 0, []

    torch.set_grad_enabled(False)
    while SMALL_EPI < MAX_EPI:
        if SMALL_EPI > 0:
            for i in range(SAMPLE_NUM):
                round_reward['each_reward'][i].append(1.0)

            env.reset()
            robot_state, inter_state, _ = env.get_dual_state()

            robot_actor_hidden = torch.zeros(env.num_robots, 32).to(device)
            robot_critic_hidden = torch.zeros(env.num_robots, 32).to(device)

            inter_critic_hidden = torch.zeros(env.num_inters, 32).to(device)

        epi_terminal = False

        while not epi_terminal:
            robot_values, robot_actions, robot_action_log_probs, robot_actor_hidden_next, robot_critic_hidden_next = \
                    generate_robot_action(robot_state, robot_actor_hidden, robot_critic_hidden, robot_network, inter_state)

            inter_actions = generate_inter_action_sample(inter_state, inter_critic_hidden, inter_network, robot_state, sampled_policy)


            robot_rewards, robot_dones, robot_infos, inter_rewards, inter_dones, inter_infos = env.step(robot_actions, inter_actions)

            robot_state_next, inter_state_next, label_target = env.get_dual_state()
            made_action = env.get_real_action(robot_actions) 


            add_one_buffer_robot(buffer, robot_state, inter_state, robot_actor_hidden, robot_critic_hidden, \
                                    robot_values, robot_actions, robot_action_log_probs,\
                                        robot_rewards, robot_dones, \
                                        label_target, made_action) 


            if len(buffer) == model_cfg.robot_buffer_length:
                robot_batch_data = transform_robot_buffer_to_batch(buffer)

                last_robot_v, _,_,_,_ = generate_robot_action(robot_state_next, robot_actor_hidden_next, robot_critic_hidden_next, robot_network, inter_state_next)

                t_batch, adv_batch = generate_robot_tar_adv_batch(robot_batch_data, last_robot_v)

                robot_batch_data['t_batch'], robot_batch_data['adv_batch'] = t_batch, adv_batch

                torch.set_grad_enabled(True)
                ppo_update_robot(global_update, robot_network, robot_batch_data)
                torch.set_grad_enabled(False)

                global_update += 1

                buffer = []

                print('global_update ', global_update)



            robot_state, inter_state = robot_state_next, inter_state_next
            robot_actor_hidden, robot_critic_hidden = robot_actor_hidden_next, robot_critic_hidden_next
            if False:
                inter_critic_hidden = inter_critic_hidden_next

            epi_terminal = True if True in robot_dones else False

            if epi_terminal:
                epi_num += 1
                SMALL_EPI += 1

            if epi_num > 0 and epi_num % model_cfg.robot_episode_save_interval == 0:
                checkpoint = {}
                checkpoint['robot_network'] = robot_network.state_dict()
                checkpoint['global_update'] = global_update
                checkpoint['epi_num'] = epi_num
                PATH = 'robot_model/' + path_prefix + '_' + str(epi_num) + '_stage2_epi_num_' + str(global_update) + '_GLOBALUPDATES' + '.checkpoint'

                torch.save(checkpoint, PATH)

                print('Save_Robot_Model: ', PATH)

    for i in range(SAMPLE_NUM):
        round_reward['avg_reward'][i] = get_avg_reward(round_reward['each_reward'][i])

    print('avg_reward: ', round_reward['avg_reward'])
    print('sampled_path: ', sampled_path)


    idxs = [ALL_PATH.index(p) for p in sampled_path]
    print('idxs: ', idxs)

    for i in range(SAMPLE_NUM):
        idx = idxs[i]
        if round_reward['avg_reward'][i] > model_cfg.sample_max_reward:
            ALL_REWARD[idx] = model_cfg.sample_max_reward

        elif round_reward['avg_reward'][i] < model_cfg.sample_min_reward:
            ALL_REWARD[idx] = model_cfg.sample_min_reward

        else:
            ALL_REWARD[idx] = round_reward['avg_reward'][i]

    print('ALL_REWARD: ', ALL_REWARD)




