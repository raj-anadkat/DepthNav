import cv2
import numpy as np
import onnxruntime as ort
import time

# Load the ONNX model with CUDAExecutionProvider
model_path = "midas_v21_small_256.onnx"
ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

cap = cv2.VideoCapture("/dev/video4")
# Set the resolution to 960x540
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Set the FPS to 60
cap.set(cv2.CAP_PROP_FPS, 60)

start_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the input frame to (256, 256)
    frame_resized = cv2.resize(frame, (256, 256))

    # Convert the input frame to a float32 numpy array
    frame_normalized = frame_resized.astype(np.float32) / 255.0

    # Transpose the input frame from (height, width, channels) to (channels, height, width)
    frame_transposed = frame_normalized.transpose((2, 0, 1))

    # Add a batch dimension to the input frame
    frame_expanded = np.expand_dims(frame_transposed, axis=0)

    # Run the ONNX model with CUDAExecutionProvider
    depth_map = ort_session.run(["output"], {"input": frame_expanded})[0][0]

    # Normalize the depth map to the range [0, 1]
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    # Resize the depth map to the same size as the input frame
    depth_map_resized = cv2.resize(depth_map_normalized, (960, 540))

    # Convert the depth map to a colored heatmap
    depth_map_colored = cv2.applyColorMap((depth_map_resized*255).astype(np.uint8), cv2.COLORMAP_JET)

    # Add the FPS text on the top left corner of the image
    fps_text = "FPS: {:.2f}".format(1.0 / (time.time() - start_time))
    cv2.putText(depth_map_colored, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the colored depth map
    cv2.imshow("Depth Map", depth_map_colored)

    # Update the start time for the next FPS calculation
    start_time = time.time()

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

