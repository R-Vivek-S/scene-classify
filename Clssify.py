from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model


app = Flask(__name__)

# load the saved model
model = load_model('model\my_model.h5')

# set the video file path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create a directory to store the extracted frames
FRAMES_FOLDER = 'frames'
if not os.path.exists(FRAMES_FOLDER):
    os.makedirs(FRAMES_FOLDER)

# define the home page


@app.route('/')
def home():
    return render_template('home.html')

# define the results page


@app.route('/predict', methods=['POST'])
def predict():
    # get the uploaded file
    video_file = request.files['video_file']

    # save the file to the uploads folder
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # read the video file
    cap = cv2.VideoCapture(video_path)

    # extract frames from the video at 2 second intervals
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 60 == 0:  # extract a frame every 2 seconds
            frame_path = os.path.join(FRAMES_FOLDER, f'frame{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
        frame_count += 1

    # loop through each extracted frame and classify it
    class_names = ['bowling', 'batting', 'boundary',
                   'closeup', 'crowd']  # replace with your class names
    results = {}
    for frame_name in os.listdir(FRAMES_FOLDER):
        frame_path = os.path.join(FRAMES_FOLDER, frame_name)
        frame = cv2.imread(frame_path)
        # resize to the input shape of your model
        frame = cv2.resize(frame, (224, 224))
        frame=frame/255.
        frame = np.expand_dims(frame, axis=0)  # add a batch dimension
        predictions = model.predict(frame)
        class_idx = np.argmax(predictions)
        class_name = class_names[class_idx]
        if class_name not in results:
            results[class_name] = []
        results[class_name].append(frame_name)

    # delete the uploaded file and extracted frames
    # os.remove(video_path)
    # for frame_name in os.listdir(FRAMES_FOLDER):
    #     frame_path = os.path.join(FRAMES_FOLDER, frame_name)
    #     os.remove(frame_path)

    # render the results page with the classification results
    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)









# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from flask import Flask, render_template, request

# # load the saved model
# model = load_model('model\my_model.h5')

# # create Flask app
# app = Flask(__name__)

# # define root route
# @app.route('/')
# def home():
#     return render_template('home.html')

# # define predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # get video file path from form data
#     video_path = request.form['video_path']

#     # read the video file
#     cap = cv2.VideoCapture(video_path)

#     # create a directory to store the extracted frames
#     frames_dir = 'frames'
#     if not os.path.exists(frames_dir):
#         os.makedirs(frames_dir)

#     # extract frames from the video at 2 second intervals
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % 60 == 0:  # extract a frame every 2 seconds
#             frame_path = os.path.join(frames_dir, f'frame{frame_count}.jpg')
#             cv2.imwrite(frame_path, frame)
#         frame_count += 1

#     # loop through each extracted frame and classify it
#     class_names = ['bowling', 'batting', 'boundary', 'closeup', 'crowd']
#     results = {}
#     for frame_name in os.listdir(frames_dir):
#         frame_path = os.path.join(frames_dir, frame_name)
#         frame = cv2.imread(frame_path)
#         # resize to the input shape of your model
#         frame = cv2.resize(frame, (224, 224))
#         frame = np.expand_dims(frame, axis=0)  # add a batch dimension
#         predictions = model.predict(frame)
#         class_idx = np.argmax(predictions[0])
#         class_name = class_names[class_idx]
#         if class_name not in results:
#             results[class_name] = []
#         results[class_name].append(frame_name)

#     # pass the results to the results.html template
#     return render_template('results.html', results=results)

# # run the app
# if __name__ == '__main__':
#     app.run(debug=True)

