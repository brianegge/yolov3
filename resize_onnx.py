import os
import sys
from pprint import pprint

import onnx


def convert(model_filename):
    out = os.path.splitext(model_filename)[0] + "_wide.onnx"
    model = onnx.load(model_filename)
    for input in model.graph.input:
        print(input.name, end=": ")
    pprint(model.graph.input[0].type.tensor_type.shape)
    tensor_type = model.graph.input[0].type.tensor_type
    for i, d in enumerate(tensor_type.shape.dim):
        if i == 0:
            d.dim_value = 1  # batch
        elif i == 2:
            d.dim_value = 384
        elif i == 3:
            d.dim_value = 672
    pprint(model.graph.input[0].type.tensor_type.shape)
    print("output")
    for output in model.graph.output:
        for i, d in enumerate(output.type.tensor_type.shape.dim):
            print("%i=" % i, end=" ")
            pprint(d)
    # model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
    # model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
    print(f"Saving to {out}")
    onnx.save(model, out)
    # self.session = onnxruntime.InferenceSession(temp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"USAGE: {sys.argv[0]} model.onnx")
    else:
        convert(sys.argv[1])
