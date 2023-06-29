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
from moving_average_filter import MovingAverageFilter

# Initialize the moving average filter with a window size of 5
scale_filter = MovingAverageFilter(window_size=5)

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
        midas_map, fps = get_midas_map(cap)
        depth_map = (1.0 / (255 - midas_map) * (1024 + 512)).astype(np.uint8)

        sensor_data = get_sensor_data(laserdev)
        for i in range(len(sensor_data)):
            sensor_data[i] = [x for x in sensor_data[i] if x < 6000]

        for i, sensor in enumerate(sensor_data):
            msg = "Sensor %i:" % i
            for dist in sensor:
                msg += "  %10i" % dist
            print(msg)

        absolute_depth_map = fuse_data(depth_map, sensor_data)
        if absolute_depth_map is None:
            continue

        # publish the laserscan message
        publish_laserscan(laserscan_pub, absolute_depth_map)

        print("FPS", fps)
        cv2.imshow('Depth Map', midas_map)

        # Wait for a key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    rclpy.shutdown()


def fuse_data(midas_map, sensor_data):
    ''' fuse_data(midas_map, [[range, ...], ...]) -> cv2.Mat

    Takes the normalized depth map and range measurements from each
    sensor and fuses them into a depth map with absolute range values.
    Returns a cv2.Map object containing absolute depth measurements.
    '''
    # img_w = 960
    # img_h = 540
    # camera_vfov = 58
    # camera_hfov = 87

    img = []
    img.append(midas_map[186:353, 0:183])
    img.append(midas_map[186:353, 183:381])
    img.append(midas_map[186:353, 381:579])
    img.append(midas_map[186:353, 579:777])
    img.append(midas_map[186:353, 777:960])

    scale_factors = []
    for i in range(0, len(img)):
        scale = calc_scale_factor(img[i], sensor_data[i], i)
        if scale is not None:
            scale_factors.append(scale)

    if len(scale_factors) == 0:
        return None

    merged_scale = merge_scale_factors(scale_factors)
    msg = ""
    for factor in scale_factors:
        msg += "%.3f  " % factor
    print("Scale factors: %s   merged: %.3f" % (msg, merged_scale))

    return midas_map * merged_scale


def calc_scale_factor(imageslice, rangelist, idx):
    ''' calc_scale_factor(imageslice, [range, ...]) -> float

    Takes the segment 'imageslice' of a normalized depth map and
    determines a floating point scaling factor to convert the
    normalized distances to absolute distances.
    '''
    if len(rangelist) == 0:
        return None

    # Calculate the histogram of the image slice
    hist = cv2.calcHist([imageslice], [0], None, [256], [0, 256])

    # Apply gaussian filter
    hist = hist[:, 0]
    hist = np.convolve(hist,
                       np.array([1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]),
                       mode='valid')

    # Find the peak of the histogram
    peak = np.argmax(hist)

    # convert pythonlist to numpy array
    rangelist = np.array(rangelist)

    # Find the minimum value of the range list
    min_range = np.min(rangelist)

    # if the peak or the min_range is 0, then the scaling factor is 0
    if (peak <= 10 or peak > 245) or min_range == 0:
        return None

    # Calculate the scaling factor
    print(idx, min_range, peak)
    scaling_factor = min_range / peak

    # filter the scaling factor
    scale_filter.update(scaling_factor)
    filtered_scale_factor = scale_filter.get_average()

    return filtered_scale_factor


def merge_scale_factors(scale_list):
    ''' merge_scale_factors([scale, ...]) -> float

    Intelligently determine the global scaling factor for the depth
    map based on the list of image segment scaling factors provided.
    '''
    # skip any scale factors that are 0
    scale_list = [x for x in scale_list if x != 0]
    merged_scale = np.mean(scale_list)

    return merged_scale


def get_midas_map(cap, buffer_size=5):
    ''' get_midas_map(cap, buffer_size) -> cv2.Mat

    Returns the greyscale normalized depth map as an OpenCV Mat.
    'device' is an cv2.VideoCapture device used as an input to the
    depth inference algorithm.
    '''
    # Create a buffer to store previous depth maps
    depth_buffer = []

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

    depth = output_data[0]

    # Invert the depth map (change this for a different convention)
    depth = np.max(depth) - depth

    depth_map = cv2.normalize(
        depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    depth_map = cv2.resize(depth_map, (960, 540))

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time

    # 0505
    # Add current depth map to the buffer
    depth_buffer.append(depth_map)

    # Apply temporal smoothing to the depth buffer
    if len(depth_buffer) > 1:
        smoothed_depth_map = np.mean(depth_buffer, axis=0)
    else:
        smoothed_depth_map = depth_map

    # If the buffer size exceeds the specified limit, remove the oldest depth map
    if len(depth_buffer) > buffer_size:
        depth_buffer.pop(0)

    return smoothed_depth_map, fps


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


def publish_laserscan(publisher, depth_map):
    ''' publish_laserscan(depth_map):

    Publishes the ROS LaserScan message using the scaled depth map data.
    'depth_map' is a cv2.Mat object.
    '''

    # Create a LaserScan message to publish absolute_depth_map
    laserscan = LaserScan()

    fov_rad = (87.0 * math.pi / 180)
    image_width = 960

    laserscan = sensor_msgs.msg.LaserScan()
    laserscan.header.stamp = rclpy.time.Time().to_msg()
    laserscan.header.frame_id = "laser_frame"
    laserscan.angle_min = -fov_rad / 2
    laserscan.angle_max = fov_rad / 2
    laserscan.angle_increment = fov_rad / image_width
    laserscan.range_min = 0.0
    laserscan.range_max = 6.0

    # get center row of depth map of size 540x960
    depth_map = depth_map[270, :]
    depth_map = depth_map / 1000     # convert mm to meters
    laserscan.ranges = list(reversed(depth_map.flatten().tolist()))
    publisher.publish(laserscan)


if __name__ == "__main__":
    main()
