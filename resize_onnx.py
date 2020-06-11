import os
import sys
import onnxruntime
from onnx.tools import update_model_dims
import onnx
import numpy as np
from pprint import pprint

def convert(model_filename):
    out = os.path.splitext(model_filename)[0] + '_wide.onnx'
    model = onnx.load(model_filename)
    for input in model.graph.input:
        print (input.name, end=": ")
    pprint(model.graph.input[0].type.tensor_type.shape)
    tensor_type = model.graph.input[0].type.tensor_type
    for i,d in enumerate(tensor_type.shape.dim):
        if i == 0:
            d.dim_value = 1 # batch
        elif i == 2:
            d.dim_value = 384
        elif i == 3:
            d.dim_value = 672
    pprint(model.graph.input[0].type.tensor_type.shape)
    print('output')
    for output in model.graph.output:
        for i,d in enumerate(output.type.tensor_type.shape.dim):
            print('%i=' % i, end=" " )
            pprint(d)
    #model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
    #model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
    print("Saving to %s" % out)
    onnx.save(model, out)
    #self.session = onnxruntime.InferenceSession(temp)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: {} modex'.format(sys.argv[0]))
    else:
        convert(sys.argv[1])
