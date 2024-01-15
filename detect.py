'''
*
* Year: 2023
* Author: Dlamini
*
'''

import os
import cv2
import torch
import PySpin
import tempfile
import datetime 
import torch.backends.cudnn as cudnn 

from numpy import random 

from models.experimental import attempt_load
from utils.plots import plot_one_box 
from utils.datasets import LoadImages
from utils.torch_utils import select_device, time_synchronized 
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging  

# Load the PySpin module 
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

# Check for cameras
num_cams = cam_list.GetSize()
if num_cams == 0:
    raise Exception("No cameras found.")

# Get camera by index 
cam = cam_list.GetByIndex(0)
cam.Init()

output_directory1 = "./runX/captured_images"
if not os.path.exists(output_directory1):
    os.makedirs(output_directory1)

output_directory2 = "./runX/detected_images"
if not os.path.exists(output_directory2):
    os.makedirs(output_directory2)

# Acquisition controls 
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureTime.SetValue(14000)
cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(32.0)

# Trigger controls
cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
cam.TriggerDelay.SetValue(68)

# Analog controls 
cam.Gain.SetValue(19)
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
cam.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All)
cam.BlackLevel.SetValue(1.5)
cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
cam.BalanceRatio.SetValue(1.5)
cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
cam.GammaEnable.SetValue(True)
cam.Gamma.SetValue(0.8)

# Image Recognition 
def detect(weights, model, source, conf_thres, iou_thres, img_size):

    # Load images 
    cudnn.benchmark = True 
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # get names and colors 
    names = model.module.names if hasattr(model, 'module') else model.names 
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference 
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]
        
        # Inference 
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        output_file = open("labels.txt", "a")

        # Process detection
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() # Detection per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # Add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    #inf = f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS'
                    inf = f'Inference ({(1E3 * (t2 - t1)):.1f}ms), NMS ({(1E3 * (t3 - t2)):.1f}ms)'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=4)
                    #plot_one_box(xyxy, im0, label=label, color=(0, 0, 128), line_thickness=3)
                    print(label, "       ", inf, end='\n')

                    timestamp =datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    output_file.write(f'{timestamp} - {label}\n')

                    # :::::::::::::::::::: comment not to save :::::::::::::::::::
                    #image_file = f"detected_image_{image_count}.jpg"
                    #image_file = os.path.join(output_directory2, image_file)
                    #cv2.imwrite(image_file, im0)
                    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        output_file.close()

# Hyper parameters
weights = 'yolov7-tiny.pt'
conf_thres = 0.7
iou_thres = 0.8        
img_size = 640

# Model initialization
set_logging()
device = select_device('')
half = device.type != 'cpu' # Half precision only supported on CUDA
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(img_size, s=stride)
if half:
    model.half()
model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


# Device controls
vendor_name = cam.DeviceVendorName.GetValue()
model_name = cam.DeviceModelName.GetValue()
print(f"{vendor_name}, {model_name}")

cam.BeginAcquisition()
print("Camera started acquirirng images.")

try: 
    image_count = 0
    capture_image_path = None

    temperature = cam.DeviceTemperature.GetValue()
    print(f"Camera sensor temperature is {temperature:.2f}°C")

    # Wait for trigger signal to activate capture
    while True:
        # Wait for the trigger signal (e.g. GPIO input) to indicate image capture
        # Once the trigger signal is received, process with image capture 

        # Retrieve the next image from the camera
        image_result = cam.GetNextImage()

        if image_result.IsIncomplete():
            print("Image incomplete eith status %d." % image_result.GetImageStatus())
        else:
            # Convert image result to a numpy array
            image_data = image_result.GetNDArray()
            
            # Convert numpy array to an RGB image  
            colored_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)

            # Save the image as a temporary file 
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                capture_image_path = temp_file.name 
                cv2.imwrite(capture_image_path, colored_image)

            image_result.Release()

            # Save capture image locally 
            # ::::::::::::: comment not to save :::::::::::::
            #image_filename = f"captured_image_{image_count}.jpg"
            #image_filename = os.path.join(output_directory1, image_filename)
            #cv2.imwrite(image_filename, colored_image)
            image_count += 1 
            # ::::::::::::::::::::::::::::::::::::::::::::::

            # Image recognition CallBack function
            if capture_image_path is not None:
                source = capture_image_path
                detect(weights, model, source, conf_thres, iou_thres, img_size)

        #key = cv2.waitKey(1) & 0xFF
        #if key == ord('q') or key == ord('Q'):
        #    break

except KeyboardInterrupt:
    pass

cam.EndAcquisition()
print("Image acquisition stopped.")
    
cam.DeInit()
print("Camera deinitialized.")

del cam
cam_list.Clear()
system.ReleaseInstance()
cv2.destroyAllWindows()


