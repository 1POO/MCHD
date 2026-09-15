dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_charadesSTA
exp_id=exp

######## data paths
train_path=../lddetr/data/charades/charades_train_release.jsonl
eval_path=../lddetr/data/charades/charades_test_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../lddetr/features/charades

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip/)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=8
eval_bsz=8
lr_drop=400
VTC_loss_coef=0.6
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4
seed=$RANDOM

PYTHONPATH=$PYTHONPATH:. python ../lddetr\\ld_detr\\train.py \
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
--lr 0.0001 \
--clip_length 1 \
--lr_drop ${lr_drop} \
--n_epoch 100 \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 1.5 \
--eval_bsz ${eval_bsz} \
--eval_epoch 1 \
--seed ${seed} \
${@:1}
