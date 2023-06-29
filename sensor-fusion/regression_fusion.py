#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import laserarray
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import rclpy
import rclpy.node
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg


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
input_size = trt.volume(engine.get_binding_shape(0)) * \
    engine.max_batch_size * np.dtype(np.float32).itemsize
output_size = trt.volume(engine.get_binding_shape(
    1)) * engine.max_batch_size * np.dtype(np.float32).itemsize
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

# Create a CUDA stream to run inference asynchronously
stream = cuda.Stream()

class LaserScanPublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('laser_scan_publisher')
        self.pub = self.create_publisher(LaserScan, 'laser_depth_map', 10)

    def publish(self, laserscan):
        self.pub.publish(laserscan)


def main():
    cap = cv2.VideoCapture("/dev/video4")
    cap.set(cv2.CAP_PROP_FPS, 60)

    laserdev = laserarray.LaserArray("/dev/laserarray")
    for i in range(0, 5):
        laserdev.enable_sensor(i)

    # create a publisher for the laserscan message
    rclpy.init()
    laserscan_pub = LaserScanPublisher()

    while True:
        inv_pred_depth,depth_map, fps = get_midas_map(cap)
        
        sensor_data = get_sensor_data(laserdev)
        for i in range(len(sensor_data)):
            sensor_data[i] = [x for x in sensor_data[i] if x < 6000]

        for i, sensor in enumerate(sensor_data):
            msg = "Sensor %i:" % i
            for dist in sensor:
                msg += "  %10i" % dist
            print(msg)

        absolute_depth = fuse_data(inv_pred_depth, sensor_data)
        if absolute_depth is None:
            continue

        # publish the laserscan message
        publish_laserscan(laserscan_pub, absolute_depth)

        print("FPS", fps)
        cv2.imshow('Depth Map', depth_map)

        # Wait for a key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    rclpy.shutdown()

# not sure of this part we have an array now, find the corresponding abs_inv_depths for pred_inv
def fuse_data(inv_pred_depth, sensor_data):
    ''' fuse_data(inv_pred_depth, [[range, ...], ...]) -> 255x255 array
    sensor_data -> actual depths    
    
    Takes the normalized depth map and range measurements from each
    sensor and fuses them into a depth map with absolute range values.
    !! Returns a cv2.Map object containing absolute depth measurements.
    '''
    #array_w = 256
    #array_h = 256
    #camera_vfov = 58
    #camera_hfov = 87

    # First invert the sensor values so we get act_inv_depth
    # find the minimum points in the region and correspond them to the array so we get 5(y,x) pairs
    # call the get_regress_coeffs to find w,b
    # 
    inv_sensor_data = 1/sensor_data

    #256: 76.8 :38.4
    #90,166

    img = []
    img.append(inv_pred_depth[90:166, 0:50])
    img.append(inv_pred_depth[90:166, 50:100])
    img.append(inv_pred_depth[90:166, 100:150])
    img.append(inv_pred_depth[90:166, 150:200])
    img.append(inv_pred_depth[90:166, 200:256])

    # finding the corresponding max values (since 1/depths)
    y = []
    x = []

    for i in range(0,len(img)):
        x.append(inv_sensor_data[i])
        y.append(np.max(img[i]))

    
    w,b = get_regress_coeff(np.array(y),np.array(x))

    abs_inv_depths = w*inv_pred_depth + b

    abs_depths = 1/abs_inv_depths

    return abs_depths


def get_midas_map(cap):
    ''' get_midas_map(cap) -> cv2.Mat

    Returns inverse predicted depth, depth_map for visualization and fps
    'device' is an cv2.VideoCapture device used as an input to the
    depth inference algorithm.
    '''
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
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Synchronize the CUDA stream and copy the output data from the GPU memory
    stream.synchronize()
    output_data = np.empty([engine.max_batch_size] +
                           list(engine.get_binding_shape(1)[1:]), dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)

    # these are the model predictions
    pred_inv_depth = output_data[0]

    # Invert the depth map (change this for a different convention)
    depth_map = np.max(pred_inv_depth) - pred_inv_depth

    depth_map = cv2.normalize(
        pred_inv_depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    depth_map = cv2.resize(depth_map, (960, 540))

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time

    return pred_inv_depth,depth_map, fps


def get_sensor_data(device):
    ''' get_sensor_data(laserarray_device) -> [[int, ...], ...]

    Returns a list containing lists ranges for each sensor on the
    platform. The outer list will have a fixed size of 5 elements, and
    the inner lists may contain between 0 and 8 range values.
    '''

    sensor_data = []
    for i in range(0, 5):
        timestamp, ranges = device.get_detections(i)
        sensor_data.append(ranges)

    return sensor_data

def get_regress_coeff(abs_inv_depth,pred_inv_depth):
    ''' get corresponding real inverted depths for the predicted inverted depths"
    abs_inv_depths : ([]) array of actual inverted sensor depth values
    pred_inv_depths : ([]) array of midas values for the corresponding points

    Returns w,b after regressing using the equation abs_inv_depth = w*pred_inv*depth + b
    '''

    w, b = np.polyfit(pred_inv_depth, abs_inv_depth, 1)


    return w,b


def publish_laserscan(publisher, depth_map):
    ''' publish_laserscan(depth_map):

    Publishes the ROS LaserScan message using the scaled depth map data.
    'depth_map' is a cv2.Mat object.
    '''

    # Create a LaserScan message to publish absolute_depth_map
    laserscan = LaserScan()

    fov_rad = (87.0 * math.pi / 180)
    image_width = 256

    laserscan = sensor_msgs.msg.LaserScan()
    laserscan.header.stamp = rclpy.time.Time().to_msg()
    laserscan.header.frame_id = "laser_frame"
    laserscan.angle_min = -fov_rad / 2
    laserscan.angle_max = fov_rad / 2
    laserscan.angle_increment = fov_rad / image_width
    laserscan.range_min = 0.0
    laserscan.range_max = 6.0

    # get center row of depth map of size 256x256
    depth_map = depth_map[128, :]
    depth_map = depth_map / 1000     # convert mm to meters
    laserscan.ranges = list(reversed(depth_map.flatten().tolist()))
    publisher.publish(laserscan)


if __name__ == "__main__":
    main()