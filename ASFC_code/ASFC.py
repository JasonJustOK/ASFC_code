import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.optim import Adam

from distributions import Categorical

from nn_modules import FOV_AUX_encoder, Multi_Head_Attention, MHA_CommBlock, Aux_Decoder

from cfgs import model_cfg

deterministic = model_cfg.deterministic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Robot_Actor_Core_p2(nn.Module):
    def __init__(self):
        super(Robot_Actor_Core_p2, self).__init__()
        cnn_out_dim = 192
        self.fov_encoder = FOV_AUX_encoder()

        self_info_len = model_cfg.self_info_len + 2
        self.self_encoder = nn.Sequential(
                nn.Linear(self_info_len, 16),
                nn.LeakyReLU(),
                nn.Linear(16, model_cfg.self_info_dim),
                nn.LeakyReLU()
            )

        self.neighbor_att = Multi_Head_Attention(model_cfg.num_heads, model_cfg.rr_qkv_in_dim, \
                                    model_cfg.rr_att_dim, model_cfg.rr_att_dim, model_cfg.rr_att_dim)

        self.project_rr_q = nn.Linear(self_info_len, model_cfg.rr_qkv_in_dim)

        self.inter_att = Multi_Head_Attention(model_cfg.num_heads, model_cfg.ri_qkv_in_dim, \
                                    model_cfg.ri_att_dim, model_cfg.ri_att_dim, model_cfg.ri_att_dim)

        self.project_ri_q = nn.Linear(self_info_len, model_cfg.ri_qkv_in_dim)

        self.mlp = nn.Sequential(
                nn.Linear(cnn_out_dim + model_cfg.self_info_dim + 2*model_cfg.rr_att_dim + 2*model_cfg.ri_att_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 192),
                nn.LeakyReLU()
            )

    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, \
                    absolute_coordinate):
        self_info = torch.cat((self_info, absolute_coordinate), dim=-1)
        batch = self_info.shape[0]

        att_q = self_info
        
        self_info_v = self.self_encoder(self_info)

        grid_map_v = self.fov_encoder(grid_map)

        neighbor_mask = neighbor_mask.unsqueeze(1)

        att_q_r = self.project_rr_q(att_q)
        att_q_r = att_q_r.unsqueeze(1)


        ally_feats, self_feats_a, _ = self.neighbor_att(att_q_r, neighbor_info, neighbor_info, neighbor_mask)
        ally_self_feats = torch.cat((ally_feats.reshape(batch, -1), self_feats_a.reshape(batch, -1)), dim=-1)

        inter_mask = inter_mask.unsqueeze(1)
        att_q_i = self.project_ri_q(att_q)
        att_q_i = att_q_i.unsqueeze(1)

        enemy_feats, self_feats_e, _ = self.inter_att(att_q_i, inter_info, inter_info, inter_mask)

        enemy_self_feats = torch.cat((enemy_feats.reshape(batch, -1), self_feats_e.reshape(batch, -1)), dim=-1)

        every_fea = torch.cat((grid_map_v, self_info_v, ally_self_feats, enemy_self_feats), dim=-1)

        every_fea = self.mlp(every_fea)

        return every_fea


class Robot_Actor_Core(nn.Module):
    def __init__(self, use_for_critic = False):
        super(Robot_Actor_Core, self).__init__()
        cnn_out_dim = 192
        self.fov_encoder = FOV_AUX_encoder()
        
        self_info_len = model_cfg.self_info_len
        self.self_encoder = nn.Sequential(
                nn.Linear(self_info_len, 16),
                nn.LeakyReLU(),
                nn.Linear(16, model_cfg.self_info_dim),
                nn.LeakyReLU()
            )

        self.neighbor_att = Multi_Head_Attention(model_cfg.num_heads, model_cfg.rr_qkv_in_dim, \
                                    model_cfg.rr_att_dim, model_cfg.rr_att_dim, model_cfg.rr_att_dim)

        self.project_rr_q = nn.Linear(self_info_len, model_cfg.rr_qkv_in_dim)

        self.inter_att = Multi_Head_Attention(model_cfg.num_heads, model_cfg.ri_qkv_in_dim, \
                                    model_cfg.ri_att_dim, model_cfg.ri_att_dim, model_cfg.ri_att_dim)

        self.project_ri_q = nn.Linear(self_info_len, model_cfg.ri_qkv_in_dim)

        self.mlp = nn.Sequential(
                nn.Linear(cnn_out_dim + model_cfg.self_info_dim + 2*model_cfg.rr_att_dim + 2*model_cfg.ri_att_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 192),
                nn.LeakyReLU()
            )

    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask):
        batch = self_info.shape[0]

        att_q = self_info
        
        self_info_v = self.self_encoder(self_info)

        grid_map_v = self.fov_encoder(grid_map)

        neighbor_mask = neighbor_mask.unsqueeze(1)

        att_q_r = self.project_rr_q(att_q)
        att_q_r = att_q_r.unsqueeze(1)


        ally_feats, self_feats_a, _ = self.neighbor_att(att_q_r, neighbor_info, neighbor_info, neighbor_mask)
        ally_self_feats = torch.cat((ally_feats.reshape(batch, -1), self_feats_a.reshape(batch, -1)), dim=-1)

        inter_mask = inter_mask.unsqueeze(1)
        att_q_i = self.project_ri_q(att_q)
        att_q_i = att_q_i.unsqueeze(1)

        enemy_feats, self_feats_e, _ = self.inter_att(att_q_i, inter_info, inter_info, inter_mask)


        enemy_self_feats = torch.cat((enemy_feats.reshape(batch, -1), self_feats_e.reshape(batch, -1)), dim=-1)


        every_fea = torch.cat((grid_map_v, self_info_v, ally_self_feats, enemy_self_feats), dim=-1)

        every_fea = self.mlp(every_fea)

        return every_fea



class Robot_Actor(nn.Module):
    def __init__(self):
        super(Robot_Actor, self).__init__()

        self.core = Robot_Actor_Core()

        self.act_fc = nn.Sequential(
                    nn.Linear(192, 32),
                    nn.LeakyReLU()
            )

        self.action_out = Categorical(32, 36)

        self.deconv = Aux_Decoder()

    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_actor_hidden):
        actor_feature = self.core(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask)
        
        actor_feature = self.act_fc(actor_feature)

        action_logits = self.action_out(actor_feature)

        actions = action_logits.mode() if deterministic else action_logits.sample()

        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs, robot_actor_hidden.detach(), action_logits


    def aux_forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_actor_hidden, \
                    made_action):
        actor_feature_trunk = self.core(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask)
        
        deconv_fea = torch.cat((actor_feature_trunk, made_action), dim = -1)
        pred_logits_map = self.deconv(deconv_fea)

        actor_feature = self.act_fc(actor_feature_trunk)

        action_logits = self.action_out(actor_feature)

        actions = action_logits.mode() if deterministic else action_logits.sample()

        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs, robot_actor_hidden.detach(), action_logits, pred_logits_map


class Robot_Critic(nn.Module):
    def __init__(self):
        super(Robot_Critic, self).__init__()

        self.core = Robot_Actor_Core_p2()

        self.global_att = MHA_CommBlock(192, 192, 1)

        self.critic_fc = nn.Sequential(
                    nn.Linear(384, 32),
                    nn.LeakyReLU(), 
                    nn.Linear(32, 1)
            )


    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_critic_hidden, \
                absolute_coordinate):

        loc = self.core(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, \
                            absolute_coordinate)

        glb = loc.unsqueeze(0)
        glb = self.global_att(glb)
        glb = glb.squeeze(0)

        loc_glb = torch.cat((loc, glb), dim=-1)

        values = self.critic_fc(loc_glb)

        return values, robot_critic_hidden.detach()




class Robot_Network(nn.Module):
    def __init__(self):
        super(Robot_Network, self).__init__()

        self.actor = Robot_Actor()

        self.critic = Robot_Critic()

        self.actor_optimizer = Adam(self.actor.parameters(), lr= 0.0005)

        self.critic_optimizer = Adam(self.critic.parameters(), lr= 0.00025)


    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state, \
                    absolute_coordinate):

        actions, action_log_probs, robot_actor_hidden, action_logits = \
                    self.actor(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state)


        values, robot_critic_hidden = self.critic(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state, \
                                            absolute_coordinate)

        return values, actions, action_log_probs, robot_actor_hidden, robot_critic_hidden


    def evaluate_actions(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state, \
                                absolute_coordinate, \
                                robot_actions, \
                                made_action):

        actions___, action_log_probs___, _, action_logits, pred_logits_map = \
                    self.actor.aux_forward(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state, \
                                            made_action)


        values, _ = self.critic(self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask, robot_hidden_state, \
                                            absolute_coordinate) 

        action_log_probs = action_logits.log_probs(robot_actions)

        dist_entropy = action_logits.entropy().mean()

        return values, action_log_probs, dist_entropy, pred_logits_map




if __name__ == '__main__':
    robot_network = Robot_Network().to(device)


    print('Total number of the parameters in Robot_Network (containing the following three parts) is ', sum(x.numel() for x in robot_network.parameters()))

    act_learn = sum(x.numel() for x in robot_network.actor.core.parameters()) + \
                sum(x.numel() for x in robot_network.actor.act_fc.parameters()) + \
                sum(x.numel() for x in robot_network.actor.action_out.parameters())

    print('Total number of the parameters for action learning is ', act_learn)

    print('Total number of the parameters for value learning is ', sum(x.numel() for x in robot_network.critic.parameters()))

    print('Total number of the parameters for auxiliary training module is ', sum(x.numel() for x in robot_network.actor.deconv.parameters()))

