import subprocess
import os

# 配置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dset_name = "tvsum"
ctx_mode = "video_sub_simple_tef"
v_feat_types = "slowfast_clip_object"
t_feat_type = "clip"
s_feat_type = "clip"

results_root = "./results-tvsum/same_weights/0.3_0.3_0.5_0.005_42"
exp_id = "exp"
ctx_mode = "video_sub_simple_tef"
# 数据路径
train_path = "./data/tvsum/tvsum_train.jsonl"
eval_path = "./data/tvsum/tvsum_val.jsonl"
eval_split_name = "val"

# 特征
feat_root = "./features/tvsum"
v_feat_dim = 0
t_feat_dim = 512
v_objectFeat_dim=0
v_feat_dirs=[]
for feat_type in ["slowfast", "object", "clip"]:
    if feat_type in v_feat_types:
        if feat_type == "slowfast":
            v_feat_dirs.append(f"{feat_root}/vid_slowfast")
            v_feat_dim += 2304
        elif feat_type == "object":
            v_feat_dirs.append(f"{feat_root}/clip_object_features")
            v_objectFeat_dim += 2560
        elif feat_type == "clip":
            v_feat_dirs.append(f"{feat_root}/vid_clip")
            v_feat_dim += 512
t_feat_dir = f"{feat_root}/query_features/"

# 字幕特征
if s_feat_type == "clip":
    s_feat_dir = f"{feat_root}/sub_features"
    s_feat_dim = 512
else:
    raise ValueError("Wrong arg for s_feat_type.")

# 训练参数
bsz = 4
# 域列表
dset_domains = ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]
# dset_domains = ["BK", "BT", "DS", "FM", "GA", "PK", "PR"]
# dset_domains = ["BT"]

# 可选：设置随机种子（可以手动设定具体值）
seed = 42
# 设置 PYTHONPATH
os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"
# 开始训练
for dset_domain in dset_domains:
    VTC_loss_coef = 0.3
    CTC_loss_coef = 0.3
    new_loss_coef = 0.5
    lr=0.005
    seed = 42

    # # 根据当前domain动态设置学习率
    # if dset_domain in ["BK"]:
    #     lr = 0.005
    #     VTC_loss_coef = 1
    #     CTC_loss_coef = 1
    #     seed = 6666
    # elif dset_domain in ["BT"]:
    #     lr = 0.005
    #     VTC_loss_coef = 1
    #     CTC_loss_coef = 1
    #     seed = 6666
    # elif dset_domain in ["DS"]:
    #     lr = 0.008
    #     VTC_loss_coef = 1
    #     CTC_loss_coef = 1
    #     seed = 6666
    # elif dset_domain in ["FM"]:
    #     lr = 0.005
    #     seed = 42
    #     VTC_loss_coef = 2
    #     CTC_loss_coef = 0.5
    # elif dset_domain in ["VU"]:
    #     lr = 0.008
    #     VTC_loss_coef = 2
    #     CTC_loss_coef = 2
    #     seed = 8888
    # elif dset_domain in ["VT"]:
    #     lr = 0.003
    #     seed = 8888
    #     VTC_loss_coef = 2
    #     CTC_loss_coef = 2
    # elif dset_domain in ["GA"]:
    #     lr = 0.005
    #     seed = 2025
    #     VTC_loss_coef = 0.5
    #     CTC_loss_coef = 0.5
    # elif dset_domain in ["PK"]:
    #     VTC_loss_coef = 1.5
    #     CTC_loss_coef = 1.5
    #     lr = 0.005
    #     seed = 2025
    # elif dset_domain in ["MS"]:
    #     lr = 0.005
    #     VTC_loss_coef = 0.5
    #     CTC_loss_coef = 0.5
    #     seed = 6666
    # elif dset_domain in ["PR"]:
    #     lr = 0.005
    #     VTC_loss_coef = 1
    #     CTC_loss_coef = 1
    #     seed = 6666
    # else:
    #     lr = 0.001
    #     VTC_loss_coef = 0.5
    #     CTC_loss_coef = 1
    #     seed = 6666

    cmd = [
            "python", "./mchd/train.py",
            "--VTC_loss_coef", str(VTC_loss_coef),
            "--CTC_loss_coef", str(CTC_loss_coef),
            "--new_loss_coef", str(new_loss_coef),
            "--dset_name", dset_name,
            "--ctx_mode", ctx_mode,
            "--train_path", train_path,
            "--eval_path", eval_path,
            "--eval_split_name", eval_split_name,
            "--v_feat_dirs", *v_feat_dirs,
            "--v_feat_dim", str(v_feat_dim),
            "--t_feat_dir", t_feat_dir,
            "--t_feat_dim", str(t_feat_dim),
            "--s_feat_dir", s_feat_dir,
            "--s_feat_dim", str(s_feat_dim),
            "--bsz", str(bsz),
            "--results_root", f"{results_root}_{dset_domain}",
            "--exp_id", exp_id,
            "--max_v_l", "1000",
            "--n_epoch", "2000",
            "--lr_drop", "2000",
            "--max_es_cnt", "-1",
            "--seed", str(seed),
            "--lr", str(lr),
            "--dset_domain", dset_domain,

        ]


        # 调用训练脚本
    print(f"Running training for domain: {dset_domain}")
    subprocess.run(cmd)
