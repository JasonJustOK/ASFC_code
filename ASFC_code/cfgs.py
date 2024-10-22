from easydict import EasyDict as edict

model_cfg = edict()
model_cfg.name = 'model_cfg'

model_cfg.deterministic = False

model_cfg.seed = 10

model_cfg.self_info_len = 4
model_cfg.self_info_dim = 8


model_cfg.num_heads = 1
model_cfg.rr_qkv_in_dim = 3
model_cfg.rr_att_dim = 64

model_cfg.ri_qkv_in_dim = 3
model_cfg.ri_att_dim = 64

model_cfg.robot_num = 8  # 5 for stage1
model_cfg.interferer_num = 6  # 5 for stage1
model_cfg.obstacle_num = 5

model_cfg.sample_init_reward = 100

model_cfg.action_interval = 0.2

model_cfg.MAX_EPI_NUM = 10000

model_cfg.robot_buffer_length = 128
model_cfg.inter_buffer_length = 128


model_cfg.robot_episode_save_interval = 200

model_cfg.GAMMA = 0.95
model_cfg.LAMDA = 0.95

model_cfg.sample_max_reward = 100
model_cfg.sample_min_reward = 20

model_cfg.EPOCH = 4
model_cfg.BATCH_SIZE = 32
model_cfg.clip_param = 0.2
model_cfg.value_loss_coef = 3.0
model_cfg.action_loss_coef = 1.0
model_cfg.entropy_coef = 1e-4
model_cfg.max_grad_norm = 0.5

model_cfg.aux_loss_coef = 30.0



