ckpt_path=./results/gcn_clip_text_simple_CLIP_slowfast/hl-video_sub_simple_tef-exp-2025_02_25_16_35_42/model_best.ckpt
eval_split_name=val
s_feat_type=clip
s_feat_dim=512
s_feat_dir=./features/qvhighlights/clip_features_s
eval_path=data/highlight_${eval_split_name}_release.jsonl

export PYTHONPATH=".:$PYTHONPATH"
PYTHONPATH=$PYTHONPATH:. python ./tr_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
${@:3}
