import math
import os
import tempfile
from pprint import pprint

import numpy as np
import onnx
import onnxruntime
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from object_detection import ObjectDetection

import common

TRT_LOGGER = trt.Logger()


class ONNXTensorRTObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""

    def __init__(self, model_filename, labels, prob_threshold=0.10, scale=4):
        super(ONNXTensorRTObjectDetection, self).__init__(labels, prob_threshold)
        # scale is the size in the input area relative to the base model size
        # so a scale of 4 means we must multiply the x and y sizes each by 2
        self.model_width = 32 * int(688 * math.sqrt(scale) / 32)
        # 1344
        self.model_height = 32 * int(384 * math.sqrt(scale) / 32)
        # 768
        engine_file_path = model_filename + ".engine"
        self.cfx = cuda.Device(0).make_context()
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        if os.path.exists(engine_file_path) and os.path.getctime(
            engine_file_path
        ) > os.path.getctime(model_filename):
            # If a serialized engine exists, use it instead of building an engine.
            print(
                "Reading engine from file {} for classes {}".format(
                    engine_file_path, ",".join(labels)
                )
            )
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            print("Compiling model {}".format(os.path.basename(model_filename)))
            self.engine = self.get_engine(model_filename, engine_file_path)
        self.is_fp16 = False  # network.get_input(0).type == 'tensor(float16)'
        self.input_name = "input"  # network.get_input(0).name
        self.context = self.engine.create_execution_context()

    def get_engine(self, onnx_file_path, engine_file_path=""):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                        onnx_file_path
                    )
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print(
                "Creating model with shape {},{}".format(
                    self.model_height, self.model_width
                )
            )
            network.get_input(0).shape = [
                1,
                3,
                self.model_height,
                self.model_width,
            ]  # NCWH
            # network.get_input(0).shape = [1, 3, 416, 416]
            print("Completed parsing of ONNX file")
            print(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            engine = builder.build_cuda_engine(network)
            if engine:
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = image.resize((self.model_width, self.model_height))
        return image

    def predict(self, preprocessed_image):
        np_image = np.array(preprocessed_image, dtype=np.float32)[
            np.newaxis, :, :, (2, 1, 0)
        ]  # RGB -> BGR
        np_image = np.ascontiguousarray(np.rollaxis(np_image, 3, 1))
        assert (
            1,
            3,
            self.model_height,
            self.model_width,
        ) == np_image.shape, "Image must be resized to model shape"

        if self.is_fp16:
            np_image = np_image.astype(np.float16)

        self.cfx.push()
        try:
            inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
            # Do inference
            inputs[0].host = np_image
            trt_outputs = common.do_inference_v2(
                self.context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        finally:
            self.cfx.pop()  # very important
        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        # There should be nothing to 'round' here. If there is, we made a mistake earlier
        output_shapes = [
            (
                1,
                (len(self.labels) + 5) * 5,
                int(self.model_height / 32),
                int(self.model_width / 32),
            )
        ]
        trt_outputs = [
            output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)
        ]
        trt_outputs = np.squeeze(trt_outputs).transpose((1, 2, 0)).astype(np.float32)
        return trt_outputs
