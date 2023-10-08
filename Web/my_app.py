# from flask import Flask
from flask import Flask, render_template,Response
import cv2

app = Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n' b'Content_Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
             
@app.route("/")
def demo():
    return render_template("demo.html")

@app.route("/video")
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    # Mở camera
    # cap = cv2.VideoCapture(0)

    # Lấy ảnh từ camera
    # stream = cv2.VideoStream(cap).start()

    # Lưu ảnh vào tệp
    # while True:
    #     # Lấy một khung từ camera
    #     ret, frame = cap.read()
    #     jpg_encoded = cv2.imencode(".jpg", frame)[1].tostring()
    #     # Xử lý khung
    #     # ...

    #     # Trả về khung
    #     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg_encoded + b'\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)