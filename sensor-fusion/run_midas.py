import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
import time

# Define the input image size
INPUT_SIZE = (256, 256)

# Define the maximum output depth value
MAX_DEPTH = 1000

# Load the TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.Severity.ERROR)
engine_file_path = "engine16.trt"
with open(engine_file_path, "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# Create a context for executing inference on the TensorRT engine
context = engine.create_execution_context()

# Allocate memory on the GPU for the input and output data
input_size = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size * np.dtype(np.float32).itemsize
output_size = trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size * np.dtype(np.float32).itemsize
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

# Create a CUDA stream to run inference asynchronously
stream = cuda.Stream()

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture("/dev/video4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,60)


## function here ##

def run_midas(cap):

    start_time = time.time()
    # Read a frame from the camera
    ret, frame = cap.read()
    # Resize the image to match the input shape of the MIDAS V21 Small 256 model
    img = cv2.resize(frame, (256, 256))

    # Convert the image to a numpy array and normalize the pixel values
    img_array = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    # Add a batch dimension to the input tensor
    input_image = img_array[np.newaxis, ...]
    

    # Copy the input data to the GPU memory
    cuda.memcpy_htod_async(d_input, input_image.ravel(), stream)

    # Execute inference on the TensorRT engine
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Synchronize the CUDA stream and copy the output data from the GPU memory
    stream.synchronize()
    output_data = np.empty([engine.max_batch_size] + list(engine.get_binding_shape(1)[1:]), dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    
    depth = output_data[0]

    # Invert the depth map (change this for a different convention)
    depth = np.max(depth) - depth

    depth_map = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    depth_map = cv2.resize(depth_map,(960,540))
    
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    
    return depth_map, fps
