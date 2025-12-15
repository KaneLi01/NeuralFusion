import argparse
import yaml
import os
import sys
import traceback
import numpy as np
from collections import defaultdict
from neural_fusion.pipelines.sculptor import SculptingPipeline
from datasets.datasets_loader.shapenetv2 import DatasetIndex
from scripts.hooks.pipeline.vis_pt_mesh_fit import PartQuadricVisHook
from utils.files.logger import write_dict_to_csv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 允许用户通过命令行指定使用哪个配置文件
    parser.add_argument("--config", type=str, 
                        default="configs/default_sculpting_all_shapes.yaml", 
                        help="Path to configuration file")

    args = parser.parse_args()

    # 1. 读取配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = load_config(args.config)
    print(cfg)

    results_buffer = defaultdict(lambda: defaultdict(list))
    all_dataset_reader = DatasetIndex(cfg['dataset_path'])
    
    for entries in all_dataset_reader.entries:
        cate = entries['category']
        model_path = entries['model_path']
        name = entries['instance']
        try:
            hooks = [PartQuadricVisHook()]
            sp = SculptingPipeline(hooks=hooks)
            metrics_dict = sp.run(cfg, model_path)
            metrics_dict['category'] = cate
            metrics_dict['instance'] = name
            write_dict_to_csv(file_path=cfg['log_path'], data_dict=metrics_dict)

        except Exception as e:
            print(f"Pipeline failed: {e}")
            traceback.print_exc()