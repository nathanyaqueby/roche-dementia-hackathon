# -*- coding: utf-8 -*-

import cv2
import face_recognition
from imutils.video import WebcamVideoStream
import os
import re
import click
import numpy as np
import textwrap

def images_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

# Encode faces from a folder
def encode_faces(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in images_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings

# Encode known images
known_names, known_face_encodings = encode_faces("C:/Users/Cfrias/Documents/hackathons/roche-dementia-hackathon/faces/")

# Threaded Video stream
#vs = WebcamVideoStream(src=0).start()
video_capture = cv2.VideoCapture(0)
assert video_capture.isOpened()  # Make sure that there is a stream.
x_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codex
out = cv2.VideoWriter("DigiMemoir.avi", four_cc, 10,
                      (x_shape, y_shape))
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #frame = vs.read()
    ret, frame = video_capture.read()
    assert ret
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
        
        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        if name == "Gaby":
            description ="Gaby is your granddaughter, she is 33 now and she loves you very much! She also loves your cookies"
            # Draw a label with a name below the face
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            wrapped_text = textwrap.wrap(description, width=30)
            x, y = 10, 40
            font_size = 0.5
            font_thickness = 1
            for i, line in enumerate(wrapped_text):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                text_w, text_h = textsize
                gap = textsize[1] + 10
                y = int((frame.shape[0] + textsize[1]) / 2) + i * gap
                x = int((frame.shape[1] - textsize[0]) / 2)
                #cv2.rectangle(frame, (x+100, y+100), (x + text_w, y + text_h), (0, 0, 0), -1)
                cv2.putText(frame, line, (x + 180, y+50), font,
                                font_size, 
                                (0,255,255), 
                                font_thickness, 
                                lineType = cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)
    out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()