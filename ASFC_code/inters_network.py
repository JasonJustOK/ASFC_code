import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.optim import Adam



from nn_modules import Multi_Head_Attention, MHA_CommBlock, FOV_AUX_encoder
from distributions import Categorical

from cfgs import model_cfg

deterministic = model_cfg.deterministic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Inter_Critic_Core(nn.Module): 
    def __init__(self):
        super(Inter_Critic_Core, self).__init__()

        cnn_out_dim = 192
        self.fov_encoder = FOV_AUX_encoder() # 64

        self_info_len = 2

        self.self_encoder = nn.Sequential(
                nn.Linear(self_info_len, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8), # 8
                nn.LeakyReLU()
            )

        self.neighbor_att = Multi_Head_Attention(1, 3, \
                                    64, 64, 64)  

        self.project_ii_q = nn.Linear(self_info_len, 3)

        self.robot_att = Multi_Head_Attention(1, 3, \
                                    64, 64, 64) 

        self.project_ir_q = nn.Linear(self_info_len, 3)

        self.mlp = nn.Sequential(
                nn.Linear(cnn_out_dim + 8 + 2*64 + 2*64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 192), # 32
                nn.LeakyReLU()
            )

    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, inter_info, inter_mask):
        batch = self_info.shape[0]

        att_q = self_info
        
        self_info_v = self.self_encoder(self_info)

        grid_map_v = self.fov_encoder(grid_map)

        neighbor_mask = neighbor_mask.unsqueeze(1)

        att_q_r = self.project_ii_q(att_q)
        att_q_r = att_q_r.unsqueeze(1)


        ally_feats, self_feats_a, _ = self.neighbor_att(att_q_r, neighbor_info, neighbor_info, neighbor_mask)

        ally_self_feats = torch.cat((ally_feats.reshape(batch, -1), self_feats_a.reshape(batch, -1)), dim=-1)


        inter_mask = inter_mask.unsqueeze(1)
        att_q_i = self.project_ir_q(att_q)
        att_q_i = att_q_i.unsqueeze(1)

        enemy_feats, self_feats_e, _ = self.robot_att(att_q_i, inter_info, inter_info, inter_mask)

        enemy_self_feats = torch.cat((enemy_feats.reshape(batch, -1), self_feats_e.reshape(batch, -1)), dim=-1)


        every_fea = torch.cat((self_info_v, grid_map_v, ally_self_feats, enemy_self_feats), dim=-1)

        every_fea = self.mlp(every_fea)


        return every_fea




class Inter_Critic(nn.Module):
    def __init__(self):
        super(Inter_Critic, self).__init__()

        self.big_embedding_core = Inter_Critic_Core() 

        self.act_fc = nn.Sequential(
                    nn.Linear(192, 32), # 32
                    nn.LeakyReLU()
            )


        self.action_out = Categorical(32, 36)

        self.critic_fc = nn.Sequential(
                    nn.Linear(192, 32),
                    nn.LeakyReLU(), 
                    nn.Linear(32, 1)
            )


    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden):

        big_embedding = self.big_embedding_core(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask)


        actor_feature = self.act_fc(big_embedding)

        action_logits = self.action_out(actor_feature)
        inter_actions = action_logits.mode() if deterministic else action_logits.sample()

        inter_action_log_probs = action_logits.log_probs(inter_actions)

        inter_values = self.critic_fc(big_embedding)


        return inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden.detach(), action_logits








class Inter_Network(nn.Module):
    def __init__(self):
        super(Inter_Network, self).__init__()

        self.actor_and_critic = Inter_Critic()

        self.optimizer = Adam(self.actor_and_critic.parameters(), lr= 0.0005)


    def forward(self, self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden):

        
        inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden, _ = \
                self.actor_and_critic(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden)


        return inter_values, inter_actions, inter_action_log_probs, inter_critic_hidden


    def evaluate_actions(self, self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden, \
                inter_actions):

        values, _, action_log_probs___, _, action_logits = \
                    self.actor_and_critic(self_info, grid_map, neighbor_info, neighbor_mask, robot_info, robot_mask, inter_critic_hidden)

        action_log_probs = action_logits.log_probs(inter_actions)

        dist_entropy = action_logits.entropy().mean()

        return values, action_log_probs, dist_entropy


