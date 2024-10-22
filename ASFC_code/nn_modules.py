import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import math

import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from torch import tanh, relu
from torch.cuda.amp import autocast



import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FOV_AUX_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=1)
        # 
        self.fc1 = nn.Linear(784, 192)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn



class Multi_Head_Attention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dout, dropout=0., bias=True):
        super(Multi_Head_Attention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Sequential(nn.Linear(n_head * d_v, n_head * d_v, bias=bias), nn.ReLU(), nn.Linear(n_head * d_v, dout, bias=bias))

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm_q = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v = nn.LayerNorm(n_head * d_v, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        residual = q

        # Transpose for attention dot product: b x n x lq x dv
        q = self.layer_norm_q(q).transpose(1, 2)
        k = self.layer_norm_k(k).transpose(1, 2)
        v = self.layer_norm_v(v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        return q, residual, attn.squeeze()



class MHA_CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MHA_CommBlock, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

    def forward(self, msg):
        # msg: 1 * N * in_dim
        num_agents = msg.size(1)
        attn_mask = np.ones((num_agents, num_agents), dtype = int)
        r, c = np.diag_indices_from(attn_mask)
        attn_mask[r, c] = 0

        attn_mask = torch.from_numpy(attn_mask).to(device)
        res = self.self_attn(msg, attn_mask)

        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # q_s: [batch_size x num_heads x num_agents x output_dim]
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # v_s: [batch_size x num_heads x num_agents x output_dim]

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1) # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (self.output_dim**0.5) # scores : [batch_size x n_heads x num_agents x num_agents]
            scores.masked_fill_(attn_mask == 0, -1e9) # Fills elements of self tensor with value where mask is one. 这里等于0才无效
            attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads*self.output_dim) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)

        return output # output: [batch_size x num_agents x output_dim]



class Aux_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(192 + 2, 784) # 注意这里是加了动作维度2
        self.dconv1 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=1)
        self.dconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.dconv3 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=3)


    def forward(self, x):
        x = F.leaky_relu(self.fc(x))

        x = x.reshape((x.size()[0], 16, 7, 7))
        x = F.leaky_relu(self.dconv1(x))
        x = F.leaky_relu(self.dconv2(x))
        x = self.dconv3(x)

        return x





