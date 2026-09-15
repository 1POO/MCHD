import subprocess
import sys
import random
import os
# 设置默认参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dset_name = 'charadesSTA'
ctx_mode = 'video_sub_simple_tef'
v_feat_types = 'slowfast_clip_object'
t_feat_type = 'clip'
s_feat_type = 'clip'
results_root = './results_charadesSTA_ablation/noktal'
exp_id = 'exp'

train_path = './data/charades/charades_train_release.jsonl'
eval_path = './data/charades/charades_test_release.jsonl'
eval_split_name = 'test'

feat_root = './features/charades'

# 处理视频特征参数
v_feat_dim = 0
v_objectFeat_dim = 0
v_feat_dirs = []
features = v_feat_types.split('_')
if 'slowfast' in features:
    v_feat_dirs.append(f'{feat_root}/slowfast')
    v_feat_dim += 2304
if 'clip' in features:
    v_feat_dirs.append(f'{feat_root}/clip_features_new/')
    v_feat_dim += 512
if 'object' in features:
    v_feat_dirs.append(f'{feat_root}/clip_object_features/')
    v_objectFeat_dim += 2560

# 处理文本特征参数
if t_feat_type == 'clip':
    t_feat_dirs = f'{feat_root}/clip_text/'
    t_feat_dim = 512
else:
    raise ValueError("Invalid t_feat_type. Must be 'clip'.")

# 处理副特征参数
if s_feat_type == 'clip':
    s_feat_dir = f"{feat_root}/sub_features"
    s_feat_dim = 512
else:
    raise ValueError("Invalid s_feat_type. Must be 'clip'.")

# 训练超参数
# bsz = 8
bsz = 8
eval_bsz = 8
lr_drop = 400
VTC_loss_coef = 0.5
CTC_loss_coef = 0.5
new_loss_coef = 0
seed = 2025

# 构建命令行参数
command = [
    'python',
    './mchd/train.py',
    '--new_loss_coef', str(new_loss_coef),
    '--VTC_loss_coef', str(VTC_loss_coef),
    '--CTC_loss_coef', str(CTC_loss_coef),
    '--dset_name', dset_name,
    '--ctx_mode', ctx_mode,
    '--train_path', train_path,
    '--eval_path', eval_path,
    '--eval_split_name', eval_split_name,
    '--v_feat_dirs'
] + v_feat_dirs + [
    '--v_feat_dim', str(v_feat_dim),
    '--t_feat_dirs', t_feat_dirs,
    '--t_feat_dim', str(t_feat_dim),
    '--s_feat_dir', s_feat_dir,
    '--s_feat_dim', str(s_feat_dim),
    '--bsz', str(bsz),
    '--results_root', results_root,
    '--exp_id', exp_id,
    '--max_v_l', '-1',
    '--lr', '0.0001',
    '--clip_length', '1',
    '--lr_drop', str(lr_drop),
    '--n_epoch', '100',
    '--contrastive_align_loss_coef', '0.002',
    '--lw_saliency', '1.5',
    '--eval_bsz', str(eval_bsz),
    '--seed', str(seed)]

# 添加用户输入的额外参数
command += sys.argv[1:]

# 执行命令
subprocess.run(command)