import cv2
import torch
from torch import hub
from time import time
import numpy as np
from imutils.video import WebcamVideoStream



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
        stream = cv2.VideoCapture(0)  # 0 means read from the default camera, 1 the next camera, and so on...
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
        vs = WebcamVideoStream(src=0).start()
        #ret, frame = player.read()
        #frame = vs.read()  # Read the first frame.
        while True:  # Run until stream is out of frames
            start_time = time()  # We would like to measure the FPS.
            frame = vs.read()
            #ret, frame = player.read()
            #assert ret
            results = self.score_frame(frame)  # Score the Frame
            frame = self.plot_boxes(results, frame)  # Plot the boxes.
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            cv2.imshow('frame', frame)
            out.write(frame)  # Write the frame onto the output.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ret, frame = player.read()  # Read next frame.
        # Release handle to the webcam
        vs.release()
        cv2.destroyAllWindows()


a = ObjectDetection()
a()
