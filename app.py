import streamlit as st
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import cv2



st.markdown(""" <style>
        .stApp{
        background: linear-gradient(to top, #CEDFF5, #FFFFFF);
            }
            
            
  div.stButton > button {
    background-color: #2F5270;
    color: white;
    font-size: 20px;
    font-weight: bold;
    width: 145px;
    height: 45px;
    border-radius: 10px;
    border: 2px double white;
    margin-top : 10px;
    margin-bottom : 10px;
}
            </style>
""",unsafe_allow_html=True)


st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #2F5270;
    color:white;
            
}       


</style>
""", unsafe_allow_html=True)


# Load the pre-trained model

model = load_model('digit_recognition_model.keras')



st.markdown(""" 
            <h1 style="color: #062136 "> ✍️ Handwritten Digit Recognition </h1>""",unsafe_allow_html=True)

st.divider()
st.markdown(""" 
            <h5 style="margin-top:20px; color:#062136; "> Draw a digit (0–9) and click Predict </h5>""",unsafe_allow_html=True)

st.sidebar.markdown(""" 
            <h1  style="color:#062136;" > <u>About </u></h1>""",unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">Model : ANN</h5>""",unsafe_allow_html=True)

st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">Dataset : MNIST</h5>""",unsafe_allow_html=True)


st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">Image Size : 28 x 28</h5>""",unsafe_allow_html=True)


st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">Framework : TensorFlow</h5>""",unsafe_allow_html=True)


st.sidebar.markdown(""" <h5 style="color:#FFFFFF;   padding:0px;">Accuracy : 97%</h5>""",unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown(""" <h3 style="color:#062136; margin-bottom:5px;">Developed by :</h3>""",unsafe_allow_html=True)
st.sidebar.markdown(""" <h5 style="color:#FFFFFF; padding:0px;">&nbsp;&nbsp;&nbsp;Vishakha Nikam </h5>""",unsafe_allow_html=True)






canvas_result = st_canvas(


    fill_color = "#00000000",  # Canvas background color -> black
    stroke_width = 10,
    stroke_color ="#FFFFFF",  # Stroke color -> white
    background_color ="#000000",
    width = 320,
    height = 320,
    drawing_mode = "freedraw",
    key = "canvas",
)

if st.button("Predict"):

    # Convert the canvas image to a numpy array
    img = canvas_result.image_data.astype(np.uint8)

    #Convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    grey_img = cv2.resize(grey_img, (28, 28))    

    # Normalize the pixel values to be between 0 and 1
    grey_img = grey_img / 255.0

    # Reshape the image to match the input shape of the model
    grey_img = grey_img.reshape(1,784)

    result = model.predict(grey_img)    # Predict the digit using the pre-trained model

    index = np.argmax(result)   # Get the index of the highest probability digit

    st.divider()
    
    st.markdown(""" 
            <h3 style="color: #062136 "><b>🎯 Predicted Digit:</b> </h3>""",unsafe_allow_html=True)
    
    st.markdown(
    f"<h2 style='  color:#062136;'> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    {index}</h2>",
    unsafe_allow_html=True
    )

    st.divider()
    
    confidence = np.max(result) * 100

    
    
    st.markdown(
    f"""
        <h3><b>✅ Confidence</b></h3>
        <h3 style="color:#1E88E5; ">
            {confidence:.2f}%
        </h3>
   
    """,
    unsafe_allow_html=True
    )
    
    st.progress(int(confidence))
    
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#A7A7A8;">
        Developed using TensorFlow, Keras and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
