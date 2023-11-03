#import libraries

from flask import Flask, render_template,Response, jsonify, request
import cv2
# from style_mediapipe import  draw_styled_landmarks,extract_keypoints, draw_lip
import mediapipe as mp
import numpy as np
# from Transformers_Landmark import get_model,predict, PreprocessLayer
import os
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# web_detect_dir = os.path.join(current_dir, 'web_detect')
sys.path.insert(0, "D:/AI/DPLS2L/S2L/Project")
import style_mediapipe
import Transformers_Landmark
# ---------------------------------- init ---------------------------------- 
model = Transformers_Landmark.get_model()
model.load_weights('./Project/weight/model.h5')
# Khởi tạo Holistic
start = Transformers_Landmark.predict(model,np.load('./Project/statics/start_up.npy'))
preprocess = Transformers_Landmark.PreprocessLayer()
all_res = []
n_frame = 0

print('Completed model loading !')
# # ---------------------------------- init ---------------------------------- 


app = Flask(__name__, static_folder='assets')
# all_res = []  

# res = ''
current_result = ""

def reset_all(res,sequence_arr):
    res += Transformers_Landmark.predict(model,sequence_arr)
    sequence=[]
    n_frame = 0
    return res,n_frame,sequence

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_face_mesh = mp.solutions.face_mesh #Hai_them
face_mesh = mp_face_mesh.FaceMesh() #Hai_them

camera = cv2.VideoCapture(0)
def generate_frames():
    global all_res
    global n_frame
    res = ''
    num_frame_space = 30
    sequence =[]
    prev_time = 0
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                current_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (current_time - prev_time)
                prev_time = current_time

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_holistic = holistic.process(frame_rgb)
                results_face_mesh = face_mesh.process(frame_rgb)
                style_mediapipe.draw_styled_landmarks(frame, results_holistic, results_face_mesh)


                n_frame +=1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                keypoints = style_mediapipe.extract_keypoints(results)
                sequence.append(keypoints)
                sequence_arr = np.array(sequence)
                if len(sequence) > num_frame_space and np.all(sequence_arr[n_frame-num_frame_space:n_frame,:84] == 0):
                    sequence_arr =  preprocess(sequence_arr).numpy()
                    __, n_frame,sequence  = reset_all(res,sequence_arr)
                    all_res.append(res)
                    res = ''
                    if (len(all_res) != 0) and (all_res[-1] !=  '. ') and (all_res[-1] != '') :
                        all_res.append('. ')

                    current_result = ''.join(all_res)
                    print('Ket qua la : ',all_res)
                    print('RESET')
                elif len(sequence) == 128:
                    res, n_frame,sequence  = reset_all(res,sequence_arr)
                    print('after 128: ',res)
                # cv2.putText(frame, ' '.join(res), (3,30), 
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {int(fps)}", (3,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    break
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
#     return frame
# def yield_live():
#     global all_res
#     global n_frame
#     if n_frame == 128:
#         yield ''.join(all_res)

# # for res in yield_live():
            
@app.route("/")
def main():
    return render_template("home.html")

@app.route("/detect")
def detect():
    return render_template("detect.html", current_result=current_result)
# @app.route("/")
# def detect():
#     return render_template("detect.html")


@app.route('/get_latest_result')
def get_latest_result():
    global all_res
    result = ''.join(all_res)
    return jsonify({'result': result})





@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run( debug=False)