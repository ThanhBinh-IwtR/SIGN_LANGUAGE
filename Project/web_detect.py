import cv2
import os
import mediapipe as mp
import numpy as np
from style_mediapipe import  draw_styled_landmarks,extract_keypoints
from Transformers_Landmark import get_model,predict
model = get_model()
model.load_weights('project/weight/model.h5')
x = np.load('project/statics/start_up.npy')
start_up = predict(model,x)
print("START UP", start_up)
# Khởi tạo Holistic
def read_file_import(uploads):
    if not os.path.exists(uploads):
        print(f"Thư mục '{uploads}' không tồn tại.")
        return
    for filename in os.listdir(uploads):
        file_path = os.path.join(uploads,filename)

        if os.path.isfile(file_path) and filename.endswith(('.mp4', '.webm','.avi')):
            print(f'Đang đọc video từ:{uploads}')
     
            cap = cv2.VideoCapture(file_path)
            sequence =  []
            res = ''
            n_frame = 0
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    # Chuyển đổi frame sang định dạng màu BGR
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Dùng Holistic để xử lý frame
                    results = holistic.process(frame_rgb)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence_arr = np.array(sequence)
                    if len(sequence) == 128:
                        res  = predict(model,sequence_arr)
                        sequence=[]
                        n_frame = 0
                        break
                    # Vẽ các landmark trên cơ thể và khuôn mặt
                    n_frame +=1
                    print(n_frame)    
            cap.release()
            cv2.destroyAllWindows()
            return(res)
        else: break

