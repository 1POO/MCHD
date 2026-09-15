#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="$PYTHONPAT."

# 基础配置
dset_name="hl"
ctx_mode="video_sub_simple_tef"
v_feat_types="slowfast_simple_clip"
t_feat_type="clip"
results_root="./new2_result"
exp_id="exp"

# 数据路径
train_path="./data/highlight_train_release.jsonl"
eval_path="./data/highlight_val_release.jsonl"
eval_split_name="val"

# 特征根目录
feat_root="./features/qvhighlights"

# 处理视频特征配置
v_feat_dirs=()
v_feat_dim=0
v_objectFeat_dim=0

# 分割特征类型
IFS='_' read -ra feat_types <<< "$v_feat_types"
for feat_type in "${feat_types[@]}"; do
  case $feat_type in
    "slowfast")
      v_feat_dirs+=("$feat_root/slowfast_features")
      ((v_feat_dim += 2304))
      ;;
    "simple_clip")
      v_feat_dirs+=("$feat_root/clip_features_object")
      ((v_objectFeat_dim += 2560))
      ;;
    "clip")
      v_feat_dirs+=("$feat_root/clip_features")
      ((v_feat_dim += 512))
      ;;
  esac
done

# 处理文本特征配置
t_feat_dirs=()
t_feat_dim=0

if [[ $t_feat_type == *"clip"* ]]; then
  t_feat_dirs+=("../extract_query_by_clip/clip_text_features")
  ((t_feat_dim += 512))
fi

if [[ $t_feat_type == *"gcn"* ]]; then
  t_feat_dirs+=("../keyBert/text_GCN_features_512_no_mean_and_reshape")
  ((t_feat_dim += 512))
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

# 构建命令参数
cmd_args=(
  "--seed" "$seed"
  "--label_loss_coef" "$label_loss_coef"
  "--VTC_loss_coef" "$VTC_loss_coef"
  "--CTC_loss_coef" "$CTC_loss_coef"
  "--dset_name" "$dset_name"
  "--ctx_mode" "$ctx_mode"
  "--train_path" "$train_path"
  "--eval_path" "$eval_path"
  "--eval_split_name" "$eval_split_name"
  "--v_feat_dim" "$v_feat_dim"
  "--v_objectFeat_dim" "$v_objectFeat_dim"
  "--t_feat_dim" "$t_feat_dim"
  "--bsz" "$bsz"
  "--results_root" "$results_root"
  "--exp_id" "$exp_id"
  "--lr" "$lr"
  "--n_epoch" "$n_epoch"
  "--lw_saliency" "$lw_saliency"
  "--lr_drop" "$lr_drop"
)

# 添加数组参数
for dir in "${v_feat_dirs[@]}"; do
  cmd_args+=("--v_feat_dirs" "$dir")
done

for dir in "${t_feat_dirs[@]}"; do
  cmd_args+=("--t_feat_dirs" "$dir")
done

# 添加用户参数
cmd_args+=("$@")

# 打印调试信息
echo "执行命令:"
echo "python ./tr_detr/train.py" "${cmd_args[@]}"

# 执行命令
python "./tr_detr/train.py" "${cmd_args[@]}"