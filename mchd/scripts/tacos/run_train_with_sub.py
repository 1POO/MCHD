import random
import subprocess
import os
import sys

# 设置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dset_name = "tacos"
ctx_mode = "video_sub_simple_tef"
v_feat_types = "slowfast_clip_object"
t_feat_type = "clip"
s_feat_type = "clip"
results_root = "./results-tacos/bs16_lr0.0001"
exp_id = "exp"
ctx_mode = "video_sub_simple_tef"

# 数据路径
train_path = "./data/tacos/train.jsonl"
eval_path = "./data/tacos/val.jsonl"
eval_split_name = "val"

# 特征路径
feat_root = "./features/tacos"

# 视频特征处理
# 视频特征
v_feat_dim = 0
v_feat_dirs = []
v_objectFeat_dim = 0
for feat_type in ["slowfast", "object", "clip"]:
    if feat_type in v_feat_types:
        if feat_type == "slowfast":
            v_feat_dirs.append(f"{feat_root}/vid_slowfast")
            v_feat_dim += 2304
        elif feat_type == "object":
            v_feat_dirs.append(f"{feat_root}/object_features")
            v_objectFeat_dim += 2560
        elif feat_type == "clip":
            v_feat_dirs.append(f"{feat_root}/vid_clip")
            v_feat_dim += 512

t_feat_dim = 0
t_feat_dirs = []
if "clip" in t_feat_type:
    t_feat_dirs.append("./features/tacos/txt_clip")
    # t_feat_dir = "/features/qvhighlights/gcn_text_features"
    # t_feat_dim = 512
    t_feat_dim += 512
elif "gcn" in t_feat_type:
    t_feat_dirs.append("./features/qvhighlights/gcn_text_features")
    t_feat_dim += 512
else:
    raise ValueError("Wrong arg for t_feat_type.")

# 字幕特征
if s_feat_type == "clip":
    s_feat_dir = f"{feat_root}/sub_features"
    s_feat_dim = 512
else:
    raise ValueError("Wrong arg for s_feat_type.")

# 训练参数
bsz = 8
lr = 0.0001
lr_drop = 200
n_epoch = 200
clip_length = 2
eval_bsz = 8
num_dummies = 50
num_prompts = 2
total_prompts = 10
VTC_loss_coef = 0.6
CTC_loss_coef = 0.5
label_loss_coef = 4
seed = 42

# 构建命令
cmd = [
    "python",
    "./mchd/train.py",
    "--seed", str(seed),
    "--label_loss_coef", str(label_loss_coef),
    "--VTC_loss_coef", str(VTC_loss_coef),
    "--CTC_loss_coef", str(CTC_loss_coef),
    "--dset_name", dset_name,
    "--ctx_mode", ctx_mode,
    "--train_path", train_path,
    "--eval_path", eval_path,
    "--eval_split_name", eval_split_name,
    "--v_feat_dirs", *v_feat_dirs,
    "--v_feat_dim", str(v_feat_dim),
    "--v_objectFeat_dim", str(v_objectFeat_dim),
    # "--t_feat_dir", t_feat_dir,
    "--t_feat_dir", "",
    "--t_feat_dirs", *t_feat_dirs,
    "--t_feat_dim", str(t_feat_dim),
    "--s_feat_dir", s_feat_dir,
    "--s_feat_dim", str(s_feat_dim),
    "--bsz", str(bsz),
    "--results_root", results_root,
    "--exp_id", exp_id,
    "--lr", str(lr),
    "--n_epoch", str(n_epoch),
    "--lr_drop", str(lr_drop),
    "--max_v_l", "-1"
]


# 添加额外参数
cmd += sys.argv[1:]

# 执行命令
subprocess.run(cmd, check=True)