import wandb
import os
import pandas as pd

api = wandb.Api()

runs = api.runs("mluo/deepcoder")

for run in runs:
    print(f"Successfully fetched run: {run.name} (ID: {run.id})")
    # 获取metrics数据并直接保存
    metrics = run.history()
    filename = f"{run.name}.csv"
    metrics.to_csv(filename, index=False)

    # 获取并打印该run的所有artifacts的key
    artifacts = run.logged_artifacts()
    print(f"\nArtifacts for run {run.name}:")
    for artifact in artifacts:
        print(f"- {artifact.name}")
        
        # 为每个artifact创建一个目录
        artifact_dir = f"{run.name}_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        # 下载artifact文件
        try:
            artifact.download(root=artifact_dir)
            print(f"  Downloaded files to {artifact_dir}:")
            # 检查单个文件是否为parquet文件
            # 遍历artifact_dir目录下的所有文件
            for root, dirs, files in os.walk(artifact_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_path = os.path.join(root, file)
                        # 读取parquet文件
                        df = pd.read_parquet(parquet_path)
                        # 构建csv文件保存路径
                        csv_path = parquet_path.replace('.parquet', '.csv')
                        # 保存为csv格式
                        df.to_csv(csv_path, index=False)
                        print(f"    Converted {parquet_path} to {csv_path}")
        except Exception as e:
            print(f"  Failed to download artifact {artifact.name}: {str(e)}")

    break