# Fine-grained semantics-driven decoupling optimization for joint video moment retrieval and highlight detection




## Supported Datasets
The relevant features have been uploaded to the corresponding cloud storage link (https://pan.quark.cn/s/a09db6662653), please click to download.

## Prerequisites

### 0. Clone and setup

```
git clone https://github.com/1POO/MCHD.git mchd
cd mchd
```

### 1. Prepare datasets
Download the features files or extract the features independently using the method described in the paper.

### 2. Install dependencies
Python version 3.10 is required. Install dependencies using:
```
pip install -r requirements.txt
```

## QVHighlights

### Training

You can train the model using only video features or both video and audio features:

```
python mchd/scripts/run_train_with_sub.py
```

The best validation accuracy is achieved at the last epoch.

### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
python mchd/scripts/run_train_with_sub.py --ckpt_path /path/to/your/checkpoint.pth  --eval_split_name val
python mchd/scripts/run_train_with_sub.py --ckpt_path /path/to/your/checkpoint.pth  --eval_split_name test 
```
For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## TVSum

### Training

Similar to QVHighlights, you can train the model on the TVSum dataset:

```
python mchd/scripts/tvsum/run_train_tvsum.py
```
## Charades-STA

### Training

Similar to QVHighlights, you can train the model on the Charades-STA dataset:

```
python mchd/scripts/charades/run_train_charades.py
```

## Citation

If you find this repository useful, please cite our work:

```
...
```
