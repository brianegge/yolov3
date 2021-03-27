import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

def transform_to_onnx(cfgfile, weightfile, onnx_file_name):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    batch_size=1
    input_names = ["input"]
    output_names = ['boxes', 'confs']
    if 'grey' in onnx_file_name:
        channels = 1
    else:
        channels = 3
    print('channels={}'.format(channels))
    x = torch.randn((batch_size, channels, model.height, model.width), requires_grad=True)
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=None)

    print('Onnx model exporting done')


if __name__ == '__main__':
    print("Converting to onnx")
    cfg_file = sys.argv[1]
    weight_file = sys.argv[2]
    onnx_file = sys.argv[3]
    onnx_path_demo = transform_to_onnx(cfg_file, weight_file, onnx_file)
