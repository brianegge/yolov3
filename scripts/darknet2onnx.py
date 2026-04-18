import sys

from tool.darknet2pytorch import Darknet
from tool.torch_utils import *


def transform_to_onnx(cfgfile, weightfile, onnx_file_name):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print(f"Loading weights from {weightfile}... Done!")
    batch_size = 1
    input_names = ["input"]
    output_names = ["boxes", "confs"]
    if "grey" in onnx_file_name:
        channels = 1
    else:
        channels = 3
    print(f"channels={channels}")
    x = torch.randn((batch_size, channels, model.height, model.width), requires_grad=True)
    torch.onnx.export(
        model,
        x,
        onnx_file_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

    print("Onnx model exporting done")


if __name__ == "__main__":
    print("Converting to onnx")
    cfg_file = sys.argv[1]
    weight_file = sys.argv[2]
    onnx_file = sys.argv[3]
    onnx_path_demo = transform_to_onnx(cfg_file, weight_file, onnx_file)
