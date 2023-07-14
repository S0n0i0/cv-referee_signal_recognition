import os

import cv2
from src.commons.utils import PATH
import pyzed.sl as sl
import numpy as np
import sys

from pathlib import Path as Pt #Avoid confusion with class path

category = "fouls"
dataset_size = 100
reinitialize_data = True

PATH.SAMPLES = os.path.join(Pt(__file__).parent.parent.parent, "samples")
PATH.DATA = os.path.join(Pt(__file__).parent.parent.parent, "data")

if not os.path.exists(PATH.DATA):
    os.makedirs(PATH.DATA)

if not os.path.exists(os.path.join(PATH.DATA, category)):
    os.makedirs(os.path.join(PATH.DATA, category))
elif reinitialize_data:
    for file in os.listdir(os.path.join(PATH.DATA, category)):
        os.remove(os.path.join(PATH.DATA, category, file))

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.svo_real_time_mode = True

# List of directories contained into PATH.SAMPLES+category
actions = os.listdir(os.path.join(PATH.SAMPLES, category))

for action in actions:
    
    if not os.path.exists(os.path.join(PATH.DATA, category, action)):
        os.makedirs(os.path.join(PATH.DATA, category, action))
    
    # List of svo files contained in action
    files = [os.path.join(PATH.SAMPLES, category, action, file) for file in os.listdir(os.path.join(PATH.SAMPLES, category, action)) if file.endswith('.svo')]

    for file in files:
        file_name = file.split('\\')[-1].split('.')[0]
        if not os.path.exists(os.path.join(PATH.DATA, category, action, file_name)):
            os.makedirs(os.path.join(PATH.DATA, category, action, file_name))

        init_params.set_from_svo_file(file)
        
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

        bodies = sl.Bodies()

        print(f'Collecting data for class {action} - file {file}')

        # analyze video with zed until it is finished
        count = 0
        keyepoints = [0]*11
        while count < dataset_size and zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies, body_runtime_param)
            for body in bodies.body_list:
                neck = np.array(body.keypoint[3:6]).flatten()
                harms = np.array(body.keypoint[10:18]).flatten()
                keyepoints = np.concatenate([harms,neck])
                # Try to train with spatial temporal positional coding
            npy_path = os.path.join(PATH.DATA, category, action, file_name, str(count))
            np.save(npy_path, keyepoints)
            count += 1
        zed.close()
        print("Finished")