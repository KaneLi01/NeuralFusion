import argparse
import yaml
import os
import sys
import traceback
from neural_fusion.pipelines.sculptor import SculptingPipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 允许用户通过命令行指定使用哪个配置文件
    parser.add_argument("--config", type=str, 
                        default="configs/default_sculpting.yaml", 
                        help="Path to configuration file")
    
    # 进阶技巧：允许通过命令行覆盖某个参数（可选）
    parser.add_argument("--data", type=str, default=None, help="Override dataset path")

    args = parser.parse_args()

    # 1. 读取配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = load_config(args.config)

    # 2. 命令行参数优先级更高 (覆盖配置文件的值)
    if args.data:
        cfg['dataset_path'] = args.data

    # 3. 运行 Pipeline
    try:
        print(cfg)
        sp = SculptingPipeline()
        sp.run(cfg)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()