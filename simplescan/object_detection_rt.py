import os
from object_detection import ObjectDetection
import onnxruntime
import onnx
import tempfile
import numpy as np
import tensorrt as trt
import common
import pycuda.driver as cuda
import pycuda.autoinit
from pprint import pprint

TRT_LOGGER = trt.Logger()

class ONNXTensorRTObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels, prob_threshold=0.10):
        super(ONNXTensorRTObjectDetection, self).__init__(labels, prob_threshold)
        engine_file_path = model_filename + ".engine"
        self.cfx = cuda.Device(0).make_context()
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        if os.path.exists(engine_file_path) and os.path.getctime(engine_file_path) > os.path.getctime(model_filename):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            self.engine = self.get_engine(model_filename, engine_file_path)
        self.is_fp16 = False # network.get_input(0).type == 'tensor(float16)'
        self.input_name = 'input' # network.get_input(0).name
        self.context = self.engine.create_execution_context()

    def get_engine(self, onnx_file_path, engine_file_path=""):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            network.get_input(0).shape = [1, 3, 768, 1376] # NCWH
            #network.get_input(0).shape = [1, 3, 416, 416]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    def predict(self, preprocessed_image):
        np_image = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        np_image = np.ascontiguousarray(np.rollaxis(np_image, 3, 1))

        if self.is_fp16:
            np_image = np_image.astype(np.float16)

        self.cfx.push()
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        # Do inference
        inputs[0].host = np_image
        trt_outputs = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        self.cfx.pop()  # very important
        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        output_shapes = [(1, (len(self.labels) + 5) * 5, 24, 43)]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        trt_outputs = np.squeeze(trt_outputs).transpose((1,2,0)).astype(np.float32)
        return trt_outputs