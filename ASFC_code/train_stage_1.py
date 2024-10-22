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


def run(path_prefix):
    env = Env()
    robot_network = Robot_Network().to(device)

    inter_network = Inter_Network().to(device)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    ###################
    env.reset()
    robot_state, inter_state, _ = env.get_dual_state()

    robot_actor_hidden = torch.zeros(env.num_robots, 32).to(device)
    robot_critic_hidden = torch.zeros(env.num_robots, 32).to(device)

    inter_critic_hidden = torch.zeros(env.num_inters, 32).to(device)
    ###################

    global_update, epi_num, buffer, global_update_inter, buffer_inter = 0, 0, [], 0, []

    torch.set_grad_enabled(False)

    while epi_num < model_cfg.MAX_EPI_NUM:
        if epi_num > 0:
            env.reset()
            robot_state, inter_state, _ = env.get_dual_state()

            robot_actor_hidden = torch.zeros(env.num_robots, 32).to(device)
            robot_critic_hidden = torch.zeros(env.num_robots, 32).to(device)

            inter_critic_hidden = torch.zeros(env.num_inters, 32).to(device)

        epi_terminal = False

        while not epi_terminal:
            robot_values, robot_actions, robot_action_log_probs, robot_actor_hidden_next, robot_critic_hidden_next = \
                    generate_robot_action(robot_state, robot_actor_hidden, robot_critic_hidden, robot_network, inter_state)
            inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden_next = \
                    generate_inter_action(inter_state, inter_critic_hidden, inter_network, robot_state)



            robot_rewards, robot_dones, robot_infos, inter_rewards, inter_dones, inter_infos = env.step(robot_actions, inter_actions)

            robot_state_next, inter_state_next, label_target = env.get_dual_state()
            made_action = env.get_real_action(robot_actions) # shape (5, 2)

            add_one_buffer_robot(buffer, robot_state, inter_state, robot_actor_hidden, robot_critic_hidden, \
                                    robot_values, robot_actions, robot_action_log_probs,\
                                        robot_rewards, robot_dones, \
                                        label_target, made_action)

            add_one_buffer_inter(buffer_inter, inter_state, robot_state, inter_critic_hidden, \
                                    inter_values, inter_actions, inter_action_log_probs, \
                                        inter_rewards, inter_dones)


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

                print('global_update: ', global_update)


            if len(buffer_inter) == model_cfg.inter_buffer_length:
                inter_batch_data = transform_inter_buffer_to_batch(buffer_inter)

                last_inter_v, _,_,_ = generate_inter_action(inter_state_next, inter_critic_hidden_next, inter_network, robot_state_next) 

                i_t_batch, i_adv_batch = generate_inter_tar_adv_batch(inter_batch_data, last_inter_v)

                inter_batch_data['t_batch'], inter_batch_data['adv_batch'] = i_t_batch, i_adv_batch

                torch.set_grad_enabled(True)
                ppo_update_inter(global_update_inter, inter_network, inter_batch_data)
                torch.set_grad_enabled(False)

                global_update_inter += 1

                buffer_inter = []


            robot_state, inter_state = robot_state_next, inter_state_next
            robot_actor_hidden, robot_critic_hidden = robot_actor_hidden_next, robot_critic_hidden_next
            inter_critic_hidden = inter_critic_hidden_next

            epi_terminal = True if True in robot_dones else False

            if epi_terminal:
                epi_num += 1


            if epi_num > 0 and epi_num % model_cfg.robot_episode_save_interval == 0:
                checkpoint = {}
                checkpoint['robot_network'] = robot_network.state_dict()
                checkpoint['global_update'] = global_update
                checkpoint['epi_num'] = epi_num
                PATH = 'robot_model/' + path_prefix + '_' + str(epi_num) + '_epi_num_' + str(global_update) + '_GLOBALUPDATES' + '.checkpoint'

                torch.save(checkpoint, PATH)

                print('Save_Robot_Model: ', PATH)


            if epi_num > 0 and epi_num % model_cfg.robot_episode_save_interval == 0:
                checkpoint = {}
                checkpoint['inter_network'] = inter_network.state_dict()
                checkpoint['global_update'] = global_update
                checkpoint['epi_num'] = epi_num
                PATH = 'inters_model/' + path_prefix + '_' + str(epi_num) + '_epi_num_' + str(global_update) + '_GLOBALUPDATES' + '.checkpoint'

                torch.save(checkpoint, PATH)

                print('Save_Inter_Model: ', PATH)



if __name__ == '__main__':
    path_prefix = sys.argv[1]
    run(path_prefix)
