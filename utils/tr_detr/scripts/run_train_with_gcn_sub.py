import os
import subprocess

# 设置变量
dset_name = "hl"
ctx_mode = "video_sub_simple_tef"
v_feat_types = "slowfast_simclip"
# v_feat_types = "clip"
t_feat_type = "clip_gcn"
s_feat_type = "gcn_clip"
results_root = "./results/普通文本+gcn文本 clip slowfast simple视频 普通字幕+gcn字幕"
exp_id = "exp"

# 数据路径
train_path = "./data/highlight_train_release.jsonl"
eval_path = "./data/highlight_val_release.jsonl"
eval_split_name = "val"

# 设置视频和文本特征路径
feat_root = "./features/qvhighlights"

# 视频特征
v_feat_dim = 0
v_feat_dirs = []
if "slowfast" in v_feat_types:
    v_feat_dirs.append(os.path.join(feat_root, "slowfast_features"))
    v_feat_dim += 2304
if "simclip" in v_feat_types:
    v_feat_dirs.append(os.path.join(feat_root, "clip_features_simple"))
    v_feat_dim += 2560
if "clip" in v_feat_types:
    v_feat_dirs.append(os.path.join(feat_root, "clip_features"))
    v_feat_dim += 512

# 文本特征
t_feat_dim = 0
t_feat_dirs = []
if "clip" in t_feat_type:
    t_feat_dirs.append("../extract_query_by_clip/clip_text_features")
    # t_feat_dir = "/features/qvhighlights/gcn_text_features"
    # t_feat_dim = 512
    t_feat_dim += 512
if "gcn" in t_feat_type:
    t_feat_dirs.append("./features/qvhighlights/gcn_text_features")
    t_feat_dim += 512
else:
    raise ValueError("Wrong arg for t_feat_type.")

# 字幕特征
s_feat_dim = 0
s_feat_dirs = []
if "clip" in s_feat_type:
    s_feat_dirs.append("./features/qvhighlights/clip_features_s")
    s_feat_dim += 512
if "gcn" in s_feat_type:
    s_feat_dirs.append("./features/qvhighlights/sub_features_gcn")
    s_feat_dim += 512
else:
    raise ValueError("Wrong arg for s_feat_type.")

# 训练参数
bsz = 32
lr_drop = 400
lr = 0.0001
n_epoch = 200
lw_saliency = 1.0
seed = 2017
VTC_loss_coef = 0.3
CTC_loss_coef = 0.5
label_loss_coef = 4

# 设置 PYTHONPATH
os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"

# 构建命令
command = [
    "python", "./mchd/train.py",
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
    # "--t_feat_dir", t_feat_dir,
    "--t_feat_dir", "",
    "--t_feat_dirs", *t_feat_dirs,
    "--t_feat_dim", str(t_feat_dim),
    "--s_feat_dir", "",
    "--s_feat_dirs", *s_feat_dirs,
    "--s_feat_dim", str(s_feat_dim),
    "--s_feat_type", str(s_feat_type),
    "--bsz", str(bsz),
    "--results_root", results_root,
    "--exp_id", exp_id,
    "--lr", str(lr),
    "--n_epoch", str(n_epoch),
    "--lw_saliency", str(lw_saliency),
    "--lr_drop", str(lr_drop),
]

# 添加额外的参数（如果有）
import sys
if len(sys.argv) > 1:
    command.extend(sys.argv[1:])

# 打印命令
print("Running command:", " ".join(command))

# 执行命令
subprocess.run(command)