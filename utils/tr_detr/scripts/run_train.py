import os
import subprocess
import sys

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(".")

# 定义基础配置
dset_name = "hl"
ctx_mode = "video_sub_simple_tef"
v_feat_types = "slowfast_simple_clip"
t_feat_type = "clip"
results_root = "./new2_result"
exp_id = "exp"

# 数据路径配置
train_path = "./data/highlight_train_release.jsonl"
eval_path = "./data/highlight_val_release.jsonl"
eval_split_name = "val"

# 特征根目录
feat_root = "./features/qvhighlights"

# 处理视频特征配置
v_feat_dirs = []
v_feat_dim = 0
v_objectFeat_dim = 0
for feat_type in ["slowfast", "simple_clip", "clip"]:
    if feat_type in v_feat_types:
        if feat_type == "slowfast":
            v_feat_dirs.append(f"{feat_root}/slowfast_features")
            v_feat_dim += 2304
        elif feat_type == "simple_clip":
            v_feat_dirs.append(f"{feat_root}/clip_features_object")
            v_objectFeat_dim += 2560
        elif feat_type == "clip":
            v_feat_dirs.append(f"{feat_root}/clip_features")
            v_feat_dim += 512

# 处理文本特征配置
t_feat_dirs = []
t_feat_dim = 0
if "clip" in t_feat_type:
    t_feat_dirs.append("../extract_query_by_clip/clip_text_features")
    t_feat_dim += 512
if "gcn" in t_feat_type:
    t_feat_dirs.append("../keyBert/text_GCN_features_512_no_mean_and_reshape")
    t_feat_dim += 512

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

# 构建命令列表
command = [
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
    "--v_feat_dirs", *v_feat_dirs,  # 展开视频特征路径
    "--t_feat_dirs", *t_feat_dirs,  # 展开文本特征路径
    "--v_feat_dim", str(v_feat_dim),
    "--v_objectFeat_dim", str(v_objectFeat_dim),
    "--t_feat_dim", str(t_feat_dim),
    "--bsz", str(bsz),
    "--results_root", results_root,
    "--exp_id", exp_id,
    "--lr", str(lr),
    "--n_epoch", str(n_epoch),
    "--lw_saliency", str(lw_saliency),
    "--lr_drop", str(lr_drop),
]

# 添加用户输入的额外参数
command += sys.argv[1:]

# 打印完整命令（调试用）
print("执行命令:")
print(" ".join(command))

# 执行命令
subprocess.run(command, check=True)