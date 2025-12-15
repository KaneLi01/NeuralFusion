import argparse
import yaml
import os
import sys
import traceback
from neural_fusion.pipelines.sculptor import SculptingPipeline
from scripts.hooks.pipeline.vis_pt_mesh_fit import PartQuadricVisHook

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 允许用户通过命令行指定使用哪个配置文件
    parser.add_argument("--config", type=str, 
                        default="configs/default_sculpting_one_shape.yaml", 
                        help="Path to configuration file")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = load_config(args.config)

    try:
        print(cfg)
        hooks = [PartQuadricVisHook()]
        sp = SculptingPipeline(hooks=hooks)
        sp.run(cfg, cfg['dataset_path'])
    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()