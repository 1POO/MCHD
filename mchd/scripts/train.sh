dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_simple_clip
t_feat_type="clip"
results_root=./results/gcn_clip_text_simple_CLIP_slowfast_no_sub
exp_id=exp

export CUDA_VISIBLE_DEVICES=0

######## data paths
train_path=./data/highlight_train_release.jsonl
eval_path=./data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=./features/qvhighlights


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"simple_clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features_object)
  (( v_feat_dim += 2560 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# 文本特征
t_feat_dim=0
t_feat_dir=()
if [[ $t_feat_type == "clip" ]]; then
    t_feat_dirs+=("../extract_query_by_clip/clip_text_features")
#    t_feat_dirs=("../keyBert/text_GCN_features_512_no_mean_and_reshape")
    ((t_feat_dim += 512))
fi

#else
#  echo "Wrong arg for t_feat_type."
#  exit 1
#fi

#### training
bsz=32
lr_drop=400
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4

export PYTHONPATH=".:$PYTHONPATH"

PYTHONPATH=$PYTHONPATH:. python ./tr_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--t_feat_dirs ${t_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir 1 \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
${@:1}
