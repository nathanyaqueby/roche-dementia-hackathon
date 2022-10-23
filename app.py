import streamlit as st
import cv2
import torch
from torch import hub
from time import time
import numpy as np


class ObjectDetection:

    def __init__(self, out_file="testing.avi"):
        self.out_file = out_file
        self.model = hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_stream(self):
        # change the number to 0 if you only have 1 camera
        stream = cv2.VideoCapture(1)  # 0 means read from the default camera, 1 the next camera, and so on...
        return stream

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    """
    The function below orchestrates the entire operation and performs the real-time parsing for video stream.
    """

    def __call__(self):
        player = self.get_video_stream()  # Get your video stream.
        assert player.isOpened()  # Make sure that there is a stream.
        # Below code creates a new video writer object to write our
        # output stream.
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codex
        out = cv2.VideoWriter(self.out_file, four_cc, 20,
                              (x_shape, y_shape))
        ret, frame = player.read()  # Read the first frame.
        frame_window = st.image([])
        while True:  # Run until stream is out of frames
            start_time = time()  # We would like to measure the FPS.
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)  # Score the Frame
            frame = self.plot_boxes(results, frame)  # Plot the boxes.
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            # cv2.imshow('frame', frame)
            frame_window.image(frame)
            out.write(frame)  # Write the frame onto the output.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ret, frame = player.read()  # Read next frame.


###############
## Dashboard ##
###############

st.set_page_config(
    page_title="DigiMemoir - Roche Dementia Hackathon",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/nathanyaqueby/roche-dementia-hackathon',
        'Report a bug': "https://github.com/nathanyaqueby/roche-dementia-hackathon",
        'About': "# Our mission is simple. To help people with dementia remember daily objects and their loved ones,"
                 " our POC takes pictures of objects & people and stores the stories associated with them. "
                 "Whenever the person focuses on an object or person, the digital memory will start talking about it, "
                 "reminding the person of the history behind that object or person. Developed during the Roche"
                 " Dementia Hackathon Challenge by Team 4 (Women in AI and Robotics)."
    }
)
st.title("DigiMemoir")
st.markdown("Welcome to **_DigiMemoir_**, a virtual person and object recognition system that"
            " enables you to **attach memories** to any everyday item. Using AI and Augmented Reality, we help"
            " dementia & Alzheimer patients remember past experiences and sentiments.")
st.markdown("Check out our documentation ([GitHub](https://github.com/nathanyaqueby/roche-dementia-hackathon))")

st.sidebar.image("DigiMemoir.png", use_column_width=True)
st.sidebar.title("Upload a new memory")

with st.sidebar.form(key ='Form1'):
    uploaded_file = st.file_uploader("Choose an image")
    user_word = st.text_input("Enter a name", "e.g. Ada Lovelace")
    category = st.radio("Choose a category", ("Person", "Object", "Landscape"))
    description = st.text_area('Describe the memory', 'It was the best of times,'
                                                      ' the worst of times,'
                                                      ' the age of wisdom, the age of foolishness, ...')
    spec = st.checkbox('Mark as extremely special')
    submitted = st.form_submit_button(label='Submit memory ⚡')

run = st.checkbox('Run')

if submitted and not run:
    st.subheader('New memory unlocked!')
    st.image(uploaded_file)
    st.subheader(f'Meet {user_word}')
    if spec:
        st.markdown("_[Marked as extremely special]_")
    st.write(f'{description}')

# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(1)
# a = ObjectDetection()

elif run:

    width = 50
    side = max((100 - width) / 2, 0.01)

    video_file = open('DigiMemoir.mp4', 'rb')
    video_bytes = video_file.read()

    col1, container, col2 = st.columns([side, width, side], gap="medium")

    if submitted:
        col1.subheader('New memory unlocked! ✨')
        col1.image(uploaded_file)
        col1.subheader(f'Introducing: {user_word}')
        col1.write(f'{description}')

    container.video(video_bytes, start_time=0)

    col2.subheader("Special memory found! 🧠")
    col2.write("Click here to play audio description")

    audio_file = open("gaby_digimemoir.mp3", "rb")
    audio_bytes = audio_file.read()

    col2.audio(audio_bytes, format="audio/ogg", start_time=0)

    col2.subheader("Related memories")
    col2.markdown("- Banu (Person)")
    col2.markdown("- Queby (Person)")

    with col2.expander("Check out the nerd stats!"):
        col2.metric(label="Emotion", value="89%", delta="Sleepy")


    #st.write("Oops! We cannot access your webcam :(")
    #st.write("Have a cat instead:")
    #st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
else:
    st.write('Check the box above to turn on the camera and start analyzing')