ckpt_path=./results/gcn_clip_text_simple_CLIP_slowfast_no_sub/hl-video_sub_simple_tef-exp-2025_02_25_19_57_30/model_best.ckpt
eval_split_name=val
eval_path=./data/highlight_${eval_split_name}_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
export PYTHONPATH=".:$PYTHONPATH"
PYTHONPATH=$PYTHONPATH:. python ./tr_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
