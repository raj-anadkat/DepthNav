# DepthNav
DepthNav-FollowTheGap combines monocular depth estimation with range finders to recover absolute depths and leverages the powerful "Follow the Gap" technique for navigation.

## I. Goals
- Develop an autonomous driving system without LIDAR that uses existing Monocular Depth Estimation Models fused with rangefinders to percieve absolute depths.
- High Inference speed and capable of operating real-time on Jetson Xavier NX
- Ensure the system is robust to different lighting condition, temporally consistent and operate in real time.

## II. Sensor Design
The sensor design involves incorporating a Time-of-Flight (ToF) sensor array and a camera for depth perception. The ToF sensor array captures range measurements, while the camera provides visual data. The relative depths are inferred using the monocular depth estimation model. To recover absolute depths, the depth map is fused with the direct range measurements obtained from the ToF sensor array. This fusion process combines the strengths of both methods, resulting in more reliable and precise depth measurements.

## VL53L4CX Time-of-Flight Sensor Array
The sensor design incorporates the VL53L4CX, a cost-effective solution known for its excellent depth accuracy. However, it comes with a fixed 18-degree field of view (FOV). To cover the full FOV of the camera, an array of rangefinders is utilized.

The sensor data is communicated through a chain of components. The sensor communicates with a SAMD51 microcontroller via I2C protocol. The SAMD51 then transfers the data to a computer through a USB connection using the libusb library.

To ensure proper alignment and coverage, a 3D-printed mount is employed. The mount positions the rangefinders at fixed 18-degree increments, matching the FOV of the VL53L4CX. This arrangement guarantees comprehensive depth measurements across the entire FOV of the camera.
<p float="left">
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/0601c4fa-09cd-4f08-b9a3-23abbe7bdc96"alt="ROI" height="200"/>
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/a3861afb-7ba1-4225-88f5-3d4731d49b5e"alt="ROI" height="200"/>
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/e8561d44-a122-41a9-b171-bfe877fd1d3b"alt="ROI" height="200"/>
 </p>
By combining the low-cost and high-accuracy VL53L4CX sensor with an array of rangefinders, efficient data transmission via I2C and USB, and a well-designed 3D-printed mount, this sensor design provides a practical solution for achieving accurate depth perception within a fixed 18-degree FOV.

## III. Monocular Depth Estimation
Monocular Depth Estimation is a task of predicting depths of each pixel from a single RGB image, where each pixel has a normalized class value (0-1).
The choice of monocular depth estimation over stereo depth estimation offers several advantages. Firstly, it provides convenience and cost-effectiveness by requiring only a single camera instead of a stereo camera setup. This simplifies the hardware requirements and reduces overall system complexity.

Stereo depth estimation relies on triangulating points between two cameras, which can result in lower resolution in areas with fewer features or misalignment between the camera views. In contrast, monocular depth models operate solely on a single image, avoiding such limitations and ensuring consistent depth estimation across the entire image. Monocular depth models can be trained on a wider range of images, enabling them to capture diverse lighting conditions and other environmental factors. This broader training scope makes the models more robust and adaptable to varying real-world scenarios, enhancing their performance and generalization capabilities.

By leveraging monocular depth estimation, developers and researchers can benefit from a more convenient and cost-effective solution that offers improved resolution, robustness to lighting changes, and broader applicability to different environments.

## Depth Models
Initially, a custom UNet model with a 256x256x3 input size was trained on the NYU Depth V2 Dataset. Unfortunately, the accuracy of this model was found to be unsatisfactory. Additionally, it exhibited high temporal inconsistency during inference, but its performance on the Xavier NX platform was not tested.

Next, an alternative approach was attempted using a UNet with a pretrained Densenet 201 backbone model. This model achieved excellent accuracy, but the inference speed on the Jetson platform was only 4 frames per second (FPS). While the accuracy was desirable, the slow inference speed posed limitations for real-time applications.

To address these concerns, the Midas v221 small model was tested, featuring a 256x256x3 input size. This model demonstrated moderate accuracy while being lightweight and efficient. Notably, on the Xavier NX platform, it achieved an impressive inference speed of 40 FPS. The resulting depth maps, generated using the Midas v2.1 small model, are displayed below.

<p float="left">
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/331a33c9-ebee-4176-a647-e59996d1bf35"alt="ROI" height="250"/>
</p>

## Model Optimization
To optimize the above models for improved inference on Nvidia Jetson, they were first exported to ONNX format. Subsequently, the ONNX models were further optimized using TensorRT. This process aimed to leverage the power of TensorRT's optimizations and hardware acceleration capabilities, enabling faster and more efficient inference on the Nvidia Jetson platform. By optimizing the models with TensorRT, the overall inference performance on the Jetson device can be significantly enhanced, ensuring smooth and real-time depth estimation for practical applications.

## IV. Recovering Absolute Depths
The process of recovering absolute depths involves several steps. Firstly, intensity peaks are correlated and linear regression is performed to estimate the relationship between these peaks and actual depth values. Depth maps are then overlapped and analyzed using histogram techniques, enabling the identification of common depth regions and refining depth estimates. Scaling factors and biases are obtained to align depth map predictions with absolute inverse depth measurements, ensuring accurate depth representation. In the next step, depth values are adjusted based on the obtained factors and biases, aligning them with the absolute depth measurements. A sensor fusion algorithm is then applied, integrating depth information from multiple sources such as cameras and range finders. This fusion algorithm combines the adjusted depth values to create a comprehensive and accurate depth representation. Together, these steps contribute to the recovery of absolute depths, providing a robust and reliable understanding of the scene.

<p float="left">
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/bbe1f484-e010-475a-a5c8-101d75e91fda"alt="ROI" width="200"/>
</p>

## V. Follow The Gap
The process for navigating through the environment can be summarized as follows. Firstly, 2D depth scans are obtained and preprocessed. The closest point in the depth ranges array is identified. To ensure safety, a protective bubble is drawn around this closest point, setting all points inside the bubble to 0, thus defining them as "gaps" or "free space." Next, the algorithm searches for the longest consecutive sequence of non-zero elements in the ranges array, representing the maximum length "gap." Within this gap, the best goal point is determined, considering factors beyond simply the furthest point. Finally, the car is actuated to move towards the goal point by publishing an AckermannDriveStamped message to the /drive topic, facilitating navigation through the identified safe path.

<p float="left">
<img src= "https://github.com/raj-anadkat/DepthNav/assets/109377585/53d2b175-d015-4925-b6fc-96629cf6ed69"alt="ROI" width="200"/>
</p>

## Raw 2D Depth Results
https://github.com/raj-anadkat/DepthNav/assets/109377585/2d673076-3418-4ebd-83cd-64186f0c576a

## Sensor Fusion Depths Results
https://github.com/raj-anadkat/DepthNav/assets/109377585/394d241e-7ef0-4e55-9da8-ba0f00f2536e

## Integrated Follow the Gap Results
https://github.com/raj-anadkat/DepthNav/assets/109377585/9056c256-3c94-4da3-a719-b78ee5906596

https://github.com/raj-anadkat/DepthNav/assets/109377585/cc9119b0-cb6e-416d-82ba-a42595abf4df









