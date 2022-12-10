import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image,ImageFile
from io import BytesIO
import numpy as np
import cv2


class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    streamlit app that uses yolov7
    '''

    def __init__(self, ):
        self.logging_main = logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self, img_size, path_yolov7_weights, path_img_i, device_i='cpu'):
        '''
        SimpleInference_YOLOV7


        INPUTS:
        VARIABLES                    TYPE    DESCRIPTION
        1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights
        3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

        OUTPUT:
        VARIABLES                    TYPE    DESCRIPTION
        1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

        CREDIT
        1.Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        2. Steven Smiley :  https://github.com/stevensmiley1989/Full_Loop_YOLO by Steven Smiley 2022/11/24
        '''

        super().__init__(img_size, path_yolov7_weights, path_img_i, device_i=device_i)

    def main(self):
        st.title('Custom YoloV7 Object Detector')
        st.write("By: Ashutosh,Raj & Sundar")
        st.subheader(""" Upload an image and run YoloV7 on it.  
        This model was trained to detect hard hat helmet. the model fails in some cases.
        (i.e. objects too close up & too far away):\n""")

        text_i_list = []
        for i, name_i in enumerate(self.names):
            # text_i_list.append(f'id={i} \t \t name={name_i}\n')
            text_i_list.append(f'{i}: {name_i}\n')
        st.selectbox('Classes', tuple(text_i_list))
        #helmet_weight= "/Users/ashutoshkumar/Desktop/PGP-AIML/AAI-521/Module_7/yolov7/weights/best.pt"
        #object_weight= "/Users/ashutoshkumar/Desktop/PGP-AIML/AAI-521/Module_7/yolov7/weights/yolov7-tiny.pt"
        self.conf_selection = st.selectbox('Confidence Threshold', tuple([0.1, 0.25, 0.5, 0.75, 0.95]))
        #self.weight_selection = st.selectbox('Confidence Threshold', tuple([helmet_weight,object_weight]))
        self.response = requests.get(self.path_img_i)

        self.img_screen = Image.open(BytesIO(self.response.content))

        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB",
                 output_format="auto")
        st.markdown('YoloV7 on streamlit for Object detection.')
        self.im0 = np.array(self.img_screen.convert('RGB'))
        self.load_image_st()
        predictions = st.button('Detect Object on the image?')
        if predictions:
            #path_yolov7_weights = "/Users/ashutoshkumar/Desktop/PGP-AIML/AAI-521/Module_7/yolov7/weights/best.pt"
            self.predict()
            predictions = False

    def load_image_st(self):
        uploaded_img = st.file_uploader(label='Upload an image')
        if type(uploaded_img) != type(None):
            self.img_data = uploaded_img.getvalue()
            st.image(self.img_data)
            #byteImgIO = io.BytesIO()
            self.im0 = Image.open(BytesIO(self.img_data))  # .convert('RGB')
            self.im0 = np.array(self.im0)

            return self.im0
        elif type(self.im0) != type(None):
            return self.im0
        else:
            return None

    def predict(self):
        self.conf_thres = self.conf_selection

        st.write('Loading image file')
        self.load_cv2mat(self.im0)
        st.write('Object detection in progress...')
        self.inference()

        self.img_screen = Image.fromarray(self.image).convert('RGB')

        self.capt = 'DETECTED:'
        if len(self.predicted_bboxes_PascalVOC) > 0:
            for item in self.predicted_bboxes_PascalVOC:
                name = str(item[0])
                conf = str(round(100 * item[-1], 2))
                self.capt = self.capt + ' name=' + name + ' confidence=' + conf + '%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB",
                 output_format="auto")
        self.image = None


if __name__ == '__main__':
    app = Streamlit_YOLOV7()

    # INPUTS for YOLOV7
    img_size = 1056
    helmet_weight = "weights/best.pt"
    object_weight = "weights/yolov7-tiny.pt"
    weight = st.radio('select the object detection type you like to perform',('Helmet','Generic'),index=0)
    if weight == 'Helmet':
        path_yolov7_weights=helmet_weight
    else:
        path_yolov7_weights = object_weight

    #weight_selection = st.radio('Select the object detection objective',(helmet_weight,object_weight),index=0)
    #path_yolov7_weights = "/Users/ashutoshkumar/Desktop/PGP-AIML/AAI-521/Module_7/yolov7/weights/best.pt"
    #path_yolov7_weights = weight_selection
    #path_yolov7_weights = "/Users/ashutoshkumar/Desktop/PGP-AIML/AAI-521/Module_7/yolov7/weights/yolov7-tiny.pt"
    path_img_i = "https://raw.githubusercontent.com/ashuxen/Helmet_Object_detection/main/usd-logo-horizontal.png"
    # INPUTS for webapp
    app.capt = "Initial project for object detection"
    app.new_yolo_model(img_size,path_yolov7_weights, path_img_i)
    #app.conf_thres = 0.65
    app.load_model()  # Load the yolov7 model

    app.main()
