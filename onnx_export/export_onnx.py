"""
pth转onnx
"""
import argparse
import math
import numpy as np
import torch

from onnx_export.alike_onnx import ALike, configs
from alnet import ALNet

def GetArgs():
    parser = argparse.ArgumentParser(description='ALIKE model export demo.')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-n",help="The model configuration")
    parser.add_argument('--model_path', default="default", help="The model path, The default is open source model")
    parser.add_argument('--device', type=str, default='cpu', help="Running device (default: cpu).")
    parser.add_argument('--export_onnx_path', type=str, default='', help='model save path.')
    parser.add_argument('--top_k', type=int, default=2000,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.2,
                        help='Detector score thr eshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--radius', type=int, default=2,
                        help='The radius of non-maximum suppression (default: 2).')
    args = parser.parse_args()
    return args

def main():
    args = GetArgs()
    w, h = 640, 400
    model = ALike(**configs[args.model],
                  model_path=args.model_path,
                  device=args.device,
                  top_k=args.top_k,
                  radius=args.radius,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit
                  )
    model.eval()
    onnx_input = torch.rand(h,w,3)
    torch.onnx.export(
        model,
        onnx_input,
        args.export_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['keypoints','descriptors','scores','scores_map']
    )
    print("导出模型成功！")

if __name__ == '__main__':
    main()
