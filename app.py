import streamlit as st
import cv2

# Get Webcam Access
cap = cv2.VideoCapture(0)

# Initialize the Webcam
def initialize_webcam():
    global cap  # Declare the global variable
    cap = cv2.VideoCapture(0)  # Get access to the default webcam
    return cap

# Get the frame from the webcam
def get_frame():
    global cap
    ret, frame = cap.read()  # Read the frame from the webcam
    return cv2.imencode('.jpg', frame)[1].tobytes()  # Encode the frame in JPEG format

# Close the webcam
def close():
    global cap
    cap.release()

# Streamlit interface
st.title("Webcam Demo")

# Get the Webcam
if st.button("Get Webcam"):
    cap = initialize_webcam()

# Get the frame from the webcam
if st.checkbox("Show Webcam"):
    st.image(get_frame(), caption='Webcam Feed', use_column_width=True)

# Close the webcam
if st.button("Close Webcam"):
    close()