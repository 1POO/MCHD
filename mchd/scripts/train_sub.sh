#!/bin/bash

# 设置变量
dset_name="hl"
ctx_mode="video_sub_simple_tef"
v_feat_types="slowfast_simple_clip"
# v_feat_types="clip"
t_feat_type="clip_gcn"
s_feat_type="clip"
results_root=./results/gcn_clip_text_simple_CLIP_slowfast
exp_id="exp"

# 数据路径
train_path="./data/highlight_train_release.jsonl"
eval_path="./data/highlight_val_release.jsonl"
eval_split_name="val"

# 设置视频和文本特征路径
feat_root="./features/qvhighlights"

# 视频特征
v_feat_dim=0
v_feat_dirs=()
if [[ $v_feat_types == *"slowfast"* ]]; then
    v_feat_dirs+=(${feat_root}/slowfast_features)
    v_feat_dim=$((v_feat_dim + 2304))
fi
if [[ ${v_feat_types} == *"simple_clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features_simple)
  (( v_feat_dim += 2560 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# 文本特征
t_feat_dim=0
t_feat_dir=()
if [[ $t_feat_type == *"clip"* ]]; then
    t_feat_dirs+=("../extract_query_by_clip/clip_text_features")
#    t_feat_dirs=("../keyBert/text_GCN_features_512_no_mean_and_reshape")
    ((t_feat_dim += 512))
fi
if [[$t_feat_type == *"gcn"*]]; then
    t_feat_dirs+=("../keyBert/text_GCN_features_512_no_mean_and_reshape")
    t_feat_dim += 512
fi
#else
#    echo "Error: Wrong arg for t_feat_type."
#    exit 1
#fi

# 字幕特征
if [[ $s_feat_type == "clip" ]]; then
    s_feat_dir="${feat_root}/clip_features_s"
    s_feat_dim=512
else
    echo "Error: Wrong arg for s_feat_type."
    exit 1
fi

# 训练参数
bsz=32
lr_drop=400
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
label_loss_coef=4

# 设置 PYTHONPATH
export PYTHONPATH=".:${PYTHONPATH}"

# 构建命令
command=(
    "python" "./tr_detr/train.py"
    "--seed" "$seed"
    "--label_loss_coef" "$label_loss_coef"
    "--VTC_loss_coef" "$VTC_loss_coef"
    "--CTC_loss_coef" "$CTC_loss_coef"
    "--dset_name" "$dset_name"
    "--ctx_mode" "$ctx_mode"
    "--train_path" "$train_path"
    "--eval_path" "$eval_path"
    "--eval_split_name" "$eval_split_name"
    "--v_feat_dirs" "${v_feat_dirs[@]}"
    "--v_feat_dim" "$v_feat_dim"
    "--t_feat_dirs" "$t_feat_dirs"
    "--t_feat_dir" "$t_feat_dir"
    "--t_feat_dim" "$t_feat_dim"
    "--s_feat_dir" "$s_feat_dir"
    "--s_feat_dim" "$s_feat_dim"
    "--bsz" "$bsz"
    "--results_root" "$results_root"
    "--exp_id" "$exp_id"
    "--lr" "$lr"
    "--n_epoch" "$n_epoch"
    "--lw_saliency" "$lw_saliency"
    "--lr_drop" "$lr_drop"
)

# 添加额外的参数（如果有）
if [[ $# -gt 0 ]]; then
    command+=("$@")
fi

# 打印命令
echo "Running command: ${command[*]}"

# 执行命令
"${command[@]}"