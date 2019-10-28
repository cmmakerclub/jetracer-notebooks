import warnings
warnings.filterwarnings('ignore')

import torchvision
import torch
from jetbot import Robot
import time

print("loading model")
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)


print("loading floor")
model.load_state_dict(torch.load('floor.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

print("model loaded")


robot = Robot()

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera() 
#image_widget = ipywidgets.Image()

prevTime = 0
def execute4(change):
    global angle, angle_last, prevTime 
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    
    xfps = 1/(sec)
    str = "FPS : %0.1f" % xfps
    print(str)
    print("running..")
    image = change['new']
    #image = camera.value
    preprocessed = preprocess(image)
    output = model(preprocessed).detach().cpu().numpy().flatten()
    # category_index = dataset.categories.index(category_widget.value)
    xx = output[0]
    yy = output[1] 

    x = int(camera.width * (xx / 2.0 + 0.5))
    y = int(camera.height * (yy / 2.0 + 0.5))

#    prediction = image.copy()
#     prediction = cv2.circle(prediction, (x, y), 8, (255, 0, 0), 3)
#     prediction = cv2.line(prediction, (x,y), (112,224), (255,0,0), 3)
#     prediction = cv2.putText(prediction,str(xx)+","+str(yy), (x-50,y+25), cv2.FONT_HERSHEY_PLAIN, 1, 2)
#     prediction = cv2.putText(prediction,str(x)+","+str(y), (x-40,y+50), cv2.FONT_HERSHEY_PLAIN, 1, 2)


    #x_slider.value = x
    #y_slider.value = y

    if xx < 0:
        robot.left_motor.value = 0.1
        robot.right_motor.value = 0.16
    else:
        robot.left_motor.value = 0.15
        robot.right_motor.value = 0.1


#     angle = np.arctan2(x, y)
#     pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value
#     angle_last = angle
#     steering_slider.value = pid + steering_bias_slider.value

#     robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
#     robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)

#    prediction = cv2.putText(prediction,str(robot.left_motor.value)+","+str(robot.right_motor.value), (x-15,y+75), cv2.FONT_HERSHEY_PLAIN, 1, 2)
#     image_widget2.value = bgr8_to_jpeg(prediction)

# execute({'new': camera.value})
camera.observe(execute4, names='value')
