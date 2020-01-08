#!/usr/bin/env python
# coding: utf-8

# # Collision Avoidance - Live Demo
# 
# In this notebook we'll use the model we trained to detect whether the robot is ``free`` or ``blocked`` to enable a collision avoidance behavior on the robot.  
# 
# ## Load the trained model
# 
# We'll assumed that you've already downloaded the ``best_model.pth`` to your workstation as instructed in the training notebook.  Now, you should upload this model into this notebook's
# directory by using the Jupyter Lab upload tool.  Once that's finished there should be a file named ``best_model.pth`` in this notebook's directory.  
# 
# > Please make sure the file has uploaded fully before calling the next cell
# 
# Execute the code below to initialize the PyTorch model.  This should look very familiar from the training notebook.

# In[1]:


import torch
import torchvision

print('6 second alexnet')
model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
#6 seconds


# Next, load the trained weights from the ``best_model.pth`` file that you uploaded

# In[2]:


print('22 second best_model')
model.load_state_dict(torch.load('best_model.pth'))
#22 seconds


# Currently, the model weights are located on the CPU memory execute the code below to transfer to the GPU device.

# In[3]:


print('3 second cuda')
device = torch.device('cuda')
model = model.to(device)
#3 seconds


# ### Create the preprocessing function
# 
# We have now loaded our model, but there's a slight issue.  The format that we trained our model doesnt *exactly* match the format of the camera.  To do that, 
# we need to do some *preprocessing*.  This involves the following steps
# 
# 1. Convert from BGR to RGB
# 2. Convert from HWC layout to CHW layout
# 3. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
# 4. Transfer the data from CPU memory to GPU memory
# 5. Add a batch dimension

# In[4]:


print('3 second camera')
import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x
#3 seconds


# Great! We've now defined our pre-processing function which can convert images from the camera format to the neural network input format.
# 
# Now, let's start and display our camera.  You should be pretty familiar with this by now.  We'll also create a slider that will display the
# probability that the robot is blocked.

# In[5]:


print('16 second traitlets')
import traitlets
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224)
image = widgets.Image(format='jpeg', width=224, height=224)
blocked_slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
#DMF Added Next Line
gpu_slider = widgets.FloatSlider(description="GPU%", min=0.0, max=1.0, orientation='vertical')

camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)

display(widgets.HBox([image, blocked_slider, gpu_slider]))
#16 seconds


# We'll also create our robot instance which we'll need to drive the motors.

# In[6]:


from jetbot import Robot

robot = Robot()
robot.stop()


# Next, we'll create a function that will get called whenever the camera's value changes.  This function will do the following steps
# 
# 1. Pre-process the camera image
# 2. Execute the neural network
# 3. While the neural network output indicates we're blocked, we'll turn left, otherwise we go forward.

# In[7]:


import torch.nn.functional as F
import time

#DMF Added gpu_usage() section
def gpu_usage():
    """Gets the Jetson's current GPU usage fraction
    
    Returns:
        float: The current GPU usage fraction.
    """
    with open('/sys/devices/gpu.0/load', 'r') as f:
        return float(f.read().strip('\n')) / 1000.0

print('105 second define update')
def update(change):
    global blocked_slider, gpu_slider, robot
    x = change['new'] 
    x = preprocess(x)
    y = model(x)
    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    #DMF Added Next Line
    gpu_slider.value = gpu_usage()

    prob_blocked = float(y.flatten()[0])
    
    blocked_slider.value = prob_blocked
    
    if prob_blocked < 0.4:
        robot.forward(0.25)
    else:
        robot.left(0.15)
    
    time.sleep(0.001)
        
update({'new': camera.value})  # we call the function once to intialize
robot.stop()
#105 seconds


# Cool! We've created our neural network execution function, but now we need to attach it to the camera for processing. 
# 
# We accomplish that with the ``observe`` function.
# 
# > WARNING: This code will move the robot!! Please make sure your robot has clearance.  The collision avoidance should work, but the neural
# > network is only as good as the data it's trained on!

# In[8]:


print('Go!')
camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera


# Awesome! If your robot is plugged in it should now be generating new commands with each new camera frame.  Perhaps start by placing your robot on the ground and seeing what it does when it reaches an obstacle.
# 
# If you want to stop this behavior, you can unattach this callback by executing the code below.

# In[ ]:

#DMF Run catbot for X seconds and then stop. May never get here. It actually gets here.
time.sleep(300)

camera.unobserve(update, names='value')
time.sleep(0.5)
robot.stop()


# Perhaps you want the robot to run without streaming video to the browser.  You can unlink the camera as below.

# In[ ]:


# camera_link.unlink()  # don't stream to browser (will still run camera)


# To continue streaming call the following.

# In[ ]:


# camera_link.link()  # stream to browser (wont run camera)


# ### Conclusion
# 
# That's it for this live demo!  Hopefully you had some fun and your robot avoided collisions intelligently! 
# 
# If your robot wasn't avoiding collisions very well, try to spot where it fails.  The beauty is that we can collect more data for these failure scenarios
# and the robot should get even better :)
