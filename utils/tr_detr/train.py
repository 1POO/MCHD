import os
import time
import json
import pprint
import random
import numpy as np
from decorator import append
from torch.autograd import gradcheck
from tqdm import tqdm, trange
from collections import defaultdict
import sys
import os


# import visdom
# 将项目根目录添加到 Python 路径
sys.path.append(".")
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./mchd'))))
from start_end_dataset_subtitles import StartEndDataset_subtitles, prepare_batch_inputs_subtitles, start_end_collate_subtitles
from start_end_dataset_subtitles_gcn import prepare_batch_inputs_subtitles_gcn, start_end_collate_subtitles_gcn, StartEndDataset_subtitles_gcn

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import BaseOptions
from start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from mchd.start_end_dataset_audio import \
    StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from mchd.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

from loss_fun.VTCLoss import VTCLoss
from loss_fun.CTCLoss import CTC_Loss
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

# 初始化visdom
# vis = visdom.Visdom(env='Training Loss Visualization')  # 创建Visdom环境


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    torch.autograd.set_detect_anomaly(True)
    # 在train_epoch函数开头添加
    if not hasattr(train_epoch, 'loss_history'):
        train_epoch.loss_history = defaultdict(list)  # 用于存储每个loss的历史值
        train_epoch.epoch_history = []  # 用于存储epoch历史

    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    #  # 初始化Visdom窗口
    # if epoch_i == 0:
    #     vis.line(X=np.array([0]), Y=np.array([0]), win='loss',
    #             opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss'))


    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.s_feat_dir is None:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        elif "gcn" in opt.s_feat_type:
            model_inputs, targets = prepare_batch_inputs_subtitles_gcn(batch[1], opt.device, non_blocking=opt.pin_memory)
        else:
            model_inputs, targets = prepare_batch_inputs_subtitles(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()
        try:
            outputs = model(**model_inputs)
        except Exception as e:
            print("Model inputs shapes:", {k: v.shape for k, v in model_inputs.items()})
            raise e
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # 所有loss和权重矩阵相乘累加
        

        src_txt_mask,   src_vid_mask = model_inputs['src_txt_mask'], model_inputs['src_vid_mask']
        pos_mask_v =  targets['src_pos_mask_v']

        src_txt_ed, src_vid_ed =  outputs['src_txt_ed'], outputs['src_vid_ed']
        loss_align = CTC_Loss()
        loss_vid_txt_align = loss_align(src_vid_ed, src_txt_ed, pos_mask_v, src_vid_mask, src_txt_mask )

        src_vid_cls_ed = outputs['src_vid_cls_ed']
        src_txt_cls_ed = outputs['src_txt_cls_ed']
        loss_align_VTC = VTCLoss()
        loss_vid_txt_align_VTC = loss_align_VTC(src_txt_cls_ed, src_vid_cls_ed)

        losses = losses + opt.VTC_loss_coef*loss_vid_txt_align_VTC + opt.CTC_loss_coef*loss_vid_txt_align
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        # 在调用 loss.backward() 前
        with torch.autograd.set_detect_anomaly(True):
            losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))
            # 更新loss历史
            train_epoch.loss_history[k].append(float(v))

        # 更新总loss历史
        train_epoch.loss_history['loss_overall'].append(losses.item())

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break
        torch.cuda.empty_cache()  # 释放未使用的缓存

# 在每个epoch结束时绘制一次
    for k in loss_dict.keys():
        # 计算该epoch的平均loss
        avg_loss = loss_meters[k].avg
        train_epoch.loss_history[k].append(avg_loss)
        
        # 绘制曲线
        # vis.line(X=np.array([epoch_i]),
        #         Y=np.array([avg_loss]),
        #         win=k,
        #         update='append' if epoch_i > 0 else None,
        #         name=k,
        #         opts=dict(title=k, xlabel='Epoch', ylabel='Loss'))

    # 绘制总loss曲线
    avg_overall_loss = loss_meters['loss_overall'].avg
    train_epoch.loss_history['loss_overall'].append(avg_overall_loss)
    # vis.line(X=np.array([epoch_i]),
    #         Y=np.array([avg_overall_loss]),
    #         win='loss_overall',
    #         update='append' if epoch_i > 0 else None,
    #         name='loss_overall',
    #         opts=dict(title='Overall Loss', xlabel='Epoch', ylabel='Loss'))


    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    # 创建TensorBoard的SummaryWriter对象,用于记录训练日志
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    # 将超参数配置以markdown格式添加到TensorBoard中
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    # 定义训练日志的格式化字符串,包含时间、epoch和loss信息
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    # 定义评估日志的格式化字符串,包含时间、epoch、loss和评估指标信息
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    if opt.s_feat_dir is None:
        # loader每次会从dataset中随机获取一个bs的数据
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            # persistent_workers=True
        )
    elif "gcn" in opt.s_feat_type:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_subtitles_gcn,
            batch_size=opt.bsz,
            num_workers=0,
            shuffle=True,
            pin_memory=opt.pin_memory,

        )
    else:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_subtitles,
            batch_size=opt.bsz,
            num_workers=4,
            shuffle=True,
            pin_memory=False
        )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    # 使用 trange 函数创建一个进度条，表示训练的 epoch 循环
    # 从 start_epoch + 1 开始，到 opt.n_epoch 结束
    # desc 参数用于设置进度条的描述信息为 "Epoch"
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer) 
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = metrics["brief"]["MR-full-mAP"]
                
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()




def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    print('##################')
    print(opt.a_feat_dir is None)
    print(opt.a_feat_dir)
    print('##################')
    if opt.s_feat_dir is None:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dirs=opt.t_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset(**dataset_config) # ** 操作符用于将字典解包为关键字参数 得到数据
    elif "gcn" in opt.s_feat_type:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_dirs=opt.t_feat_dirs,
            s_feat_dirs=opt.s_feat_dirs,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset_subtitles_gcn(**dataset_config)
    else:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_dirs=opt.t_feat_dirs,
            s_feat_dir=opt.s_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset_subtitles(**dataset_config)


    # 把config修改为eval_dataset的config，从而获得eval_config
    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        # dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        if opt.s_feat_dir is None:
            eval_dataset = StartEndDataset(**dataset_config)
        elif "gcn" in opt.s_feat_type:
            eval_dataset = StartEndDataset_subtitles_gcn(**dataset_config)
        else:
            eval_dataset = StartEndDataset_subtitles(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model) # 计算参数量
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    # import pdb; pdb.set_trace()

    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
