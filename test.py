import torch
from Res_net import resnet18
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np

classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Comtempt']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define hyper-parameter
img_size = (192,192)

#define model
model = resnet18()
pre_trained = torch.load('./saved_models/saved_model_epoch_25.pth')
model.load_state_dict(pre_trained)

#port to model to gpu if you have gpu
model = model.to(device)
model.eval()

# load and pre-process testing image
# Note: you need to precess testing image similarly to the training images 
img_path = './test_img.jpg'
img_raw = cv2.imread(img_path)

# resize img to 48x48
img = cv2.resize(img_raw, img_size)

# convert from RGB img to gray img
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# normalize img from [0, 255] to [0, 1]
img = img/255
img = img.astype('float32')

# convert image to torch with size (1, 1, 192, 192)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)

with torch.no_grad():
    img = img.to(device)
    age_prediction = model(img)           
    age_prediction = age_prediction.data.cpu().numpy()

    # denormalize output
    age_prediction = age_prediction*100
    print('Age prediction: ', int(age_prediction))

    cv2.putText(img_raw, 'Age prediction: '+str(int(age_prediction)) + ' tuoi', (20, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    cv2.imshow('Age prediction', np.array(img_raw))
    cv2.waitKey(0)



