from django.shortcuts import render
import numpy as np
import os
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
import cv2
from django.contrib import messages


face_mask_model = load_model(r'C://Users//Mehedi//Desktop//Machine Learning Project//TumorDetection//static//model//brain-tumor-detection-model.h5')

# Create your views here.
def index(request):
    if request.method == "POST":

        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)

        input_image_path = file_url

        input_image = cv2.imread(input_image_path)

        # cv2_imshow(input_image)

        input_image_resized = cv2.resize(input_image, (128,128))

        input_image_scaled = input_image_resized/255

        input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

        input_prediction = face_mask_model.predict(input_image_reshaped)


        input_pred_label = np.argmax(input_prediction)

        print(input_pred_label)

        if input_pred_label == 1:
            messages.error(request, 'The person has brain tumor.')
            print('The person has brain tumor.')
        else:
            messages.success(request, 'The person has no brain tumor.')
            print('The person has no brain tumor.')



    return render(request, 'index.html')
