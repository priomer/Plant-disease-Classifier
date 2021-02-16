# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 21:23:16 2021

@author: ocn
"""

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor
    
model = load_model("model_all_plant_disease1.h5")
img_path = "E:/AVRN_Report/Plant_Diseases_Dataset/test/tomato_lb (2).jpg"
check_image = load_image(img_path)
prediction = model.predict(check_image)
print(prediction)

prediction =np.argmax(prediction, axis=1)
if prediction==0:
    prediction="tomato_bacterial_spot"
elif prediction==1:
    prediction="tomato_early_blight"
elif prediction==2:
    prediction="tomato_late_blight"
elif prediction==3:
    prediction="tomato_leaf_mold"
elif prediction==4:
    prediction="tomato_septoria_leaf_spot"
elif prediction==5:
    prediction="tomato_two-spotted_spider_mite"
elif prediction==6:
    prediction="tomato_target_spot"
elif prediction==7:
    prediction="tomato_yellow_Leaf_Curl_Virus"
elif prediction==8:
    prediction="tomato_mosaic_virus"
elif prediction==9:
    prediction="tomato_healthy"
elif prediction==10:
    prediction="pepper_bell_bacterial"
elif prediction==11:
    prediction="pepper_bell_healthy"
elif prediction==12:
    prediction="potato_late_blight"
elif prediction==13:
    prediction="potato_healthy"
else:
    prediction="potato_late_blight"

print(prediction)    
    
    
  
    
    
