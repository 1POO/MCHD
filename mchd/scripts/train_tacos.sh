dset_name=tacos
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results_tacos
exp_id=exp

######## data paths
train_path=../lddetr\\data\\tacos\\train.jsonl
eval_path=../lddetr\\data\\tacos\\test.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../lddetr\\features\\tacos

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=2
lr=0.00015
lr_drop=200
n_epoch=200
clip_length=2
eval_bsz=2
num_dummies=50
num_prompts=2
total_prompts=10
VTC_loss_coef=0.6
CTC_loss_coef=0.5
label_loss_coef=4
seed=$RANDOM

PYTHONPATH=$PYTHONPATH:. python ld_detr/train.py \
--label_loss_coef ${label_loss_coef} \
--VTC_loss_coef ${VTC_loss_coef} \
--CTC_loss_coef ${CTC_loss_coef} \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--max_v_l -1 \
--clip_length ${clip_length} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--contrastive_align_loss_coef 0.002 \
--eval_bsz ${eval_bsz} \
--lw_saliency 4 \
--lr 0.0001 \
--eval_epoch 1 \
--seed ${seed} \
--debug \
${@:1}
