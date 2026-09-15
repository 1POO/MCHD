import os
import sys
import subprocess

def main():
    # 硬编码固定参数
    ckpt_path = "/results/gcn_clip_text_simple_CLIP_slowfast_no_sub/model_best.ckpt"
    eval_split_name = "val"
    eval_path = f"./data/highlight_{eval_split_name}_release.jsonl"

    # 打印参数（与原Shell脚本的echo等效）
    print(f"[DEBUG] ckpt_path: {ckpt_path}")
    print(f"[DEBUG] eval_split_name: {eval_split_name}")
    print(f"[DEBUG] eval_path: {eval_path}")

    # 设置环境变量（同时包含项目路径和当前目录）
    project_path = "."
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{project_path}:{original_pythonpath}:."  # 同时添加项目路径和当前目录

    # 构建执行命令
    command = [
        "python",
        "./mchd/inference.py",
        "--resume", ckpt_path,
        "--eval_split_name", eval_split_name,
        "--eval_path", eval_path
    ]

    # 添加额外参数（支持从命令行传入）
    if len(sys.argv) > 1:
        command.extend(sys.argv[1:])  # 使用sys.argv[1:]获取所有附加参数

    # 执行命令
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 执行失败，退出码: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()