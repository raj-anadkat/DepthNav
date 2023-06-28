import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create an instance of the depth post-processing class
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# Set hole filling option to enable hole filling
hole_filling.set_option(rs.option.holes_fill, 1)

# Initialize FPS counter
fps = 0
prev_time = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Apply the post-processing filters to the depth frame
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply Levin et al. colorization to fill in missing values
        colorizer = rs.colorizer()
        colorized_depth = colorizer.colorize(depth_frame).get_data()

        # Convert colorized depth to numpy array
        colorized_depth = np.asarray(colorized_depth)

        # Calculate FPS
        curr_time = time.time()
        fps += 1
        if curr_time - prev_time > 1:
            fps_text = "FPS: {:.2f}".format(fps / (curr_time - prev_time))
            prev_time = curr_time
            fps = 0

        # Show the depth image with FPS info
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.putText(depth_colormap, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Depth Image', depth_colormap)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
