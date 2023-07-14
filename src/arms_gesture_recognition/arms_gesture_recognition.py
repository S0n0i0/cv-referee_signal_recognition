from scipy import stats
import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import os
import tensorflow as tf

from pathlib import Path as Pt #Avoid confusion with class path

from src.commons.utils import PATH

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

category = "fouls"
PATH.MODELS = os.path.join(Pt(__file__).parent.parent.parent, "models")
model_path = os.path.join(PATH.MODELS,"model_files")
category_path = os.path.join(PATH.DATA, category)

# load model_fouls.keras from model_pth
model = tf.keras.models.load_model(os.path.join(model_path, "model_fouls.keras"))
actions = np.array(os.listdir(category_path))

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Enable Positional tracking (mandatory for object detection)
positional_tracking_parameters = sl.PositionalTrackingParameters()
# If the camera is static, uncomment the following line to have better performances
# positional_tracking_parameters.set_as_static = True
zed.enable_positional_tracking(positional_tracking_parameters)

body_param = sl.BodyTrackingParameters()
body_param.enable_tracking = True                # Track people across images flow
body_param.enable_body_fitting = False            # Smooth skeleton move
body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
body_param.body_format = sl.BODY_FORMAT.BODY_70  # Choose the BODY_FORMAT you wish to use

# Enable Object Detection module
zed.enable_body_tracking(body_param)

body_runtime_param = sl.BodyTrackingRuntimeParameters()
body_runtime_param.detection_confidence_threshold = 40

# Get ZED camera information
camera_info = zed.get_camera_information()

# 2D viewer utilities
display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                , display_resolution.height / camera_info.camera_configuration.resolution.height]

# Create OpenGL viewer
viewer = gl.GLViewer()
viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)

# Create ZED objects filled in the main loop
bodies = sl.Bodies()
image = sl.Mat()

while viewer.is_available():
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        # Retrieve bodies
        zed.retrieve_bodies(bodies, body_runtime_param)

        keypoints = []
        for body in bodies.body_list:
            neck = np.array(body.keypoint[3:6]).flatten()
            harms = np.array(body.keypoint[10:18]).flatten()
            keypoints = np.concatenate([harms,neck])

        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
            #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            #image = prob_viz(res, actions, image, colors)
        
        print("Prediction: ",sentence)
        '''cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)'''
        
        # Show to screen
        #cv2.imshow('OpenCV Feed', image)

        # Update GL view
        viewer.update_view(image, bodies) 
        # Update OCV view
        image_left_ocv = image.get_data()
        cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
        cv2.imshow("ZED | 2D View", image_left_ocv)
        cv2.waitKey(25)

viewer.exit()
image.free(sl.MEM.CPU)
zed.close()