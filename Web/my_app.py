
from flask import Flask, render_template,Response, jsonify, request
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Kiểm tra xem thư mục tồn tại, nếu không thì tạo nó
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
             
@app.route("/")
def demo():
    return render_template("demo.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Không có phần tệp nào được chọn."

    file = request.files['file']

    if file.filename == '':
        return "Không có tên tệp nào được chọn."

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'message': f"Tệp {file.filename} đã được tải lên và lưu trữ trong thư mục {UPLOAD_FOLDER}."})

if __name__ == "__main__":
    app.run( debug=True)