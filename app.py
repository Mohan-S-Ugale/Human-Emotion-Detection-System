from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import pyttsx3

app = Flask(__name__)

# ================= LOAD MODEL =================

model = load_model("model/emotion_model.h5")

face_detector = cv2.CascadeClassifier(
    "model/haarcascade_frontalface_default.xml"
)

# ================= EMOTION LABELS =================

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# ================= GLOBAL VARIABLES =================

last_spoken_emotion = ""
speech_lock = False

# ================= TEXT TO SPEECH =================

def speak_emotion(emotion):
    global last_spoken_emotion, speech_lock

    # Avoid repeating same emotion
    if emotion == last_spoken_emotion:
        return

    # Prevent overlapping speech
    if speech_lock:
        return

    last_spoken_emotion = emotion
    speech_lock = True

    def speak():
        global speech_lock

        try:
            engine = pyttsx3.init()

            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)

            engine.say(f"You look {emotion}")
            engine.runAndWait()

        except Exception as e:
            print("Speech Error:", e)

        speech_lock = False

    threading.Thread(target=speak, daemon=True).start()

# ================= CAMERA =================

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    print("❌ Camera not opened")
else:
    print("✅ Camera working")

# ================= VIDEO STREAM =================

def generate_frames():

    while True:

        success, frame = video.read()

        if not success:
            continue

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # Detect emotion for each face
        for (x, y, w, h) in faces:

            # Extract face region
            roi_gray = gray[y:y + h, x:x + w]

            # Resize image
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Normalize
            roi = roi_gray.astype('float') / 255.0

            # Reshape for CNN
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Predict emotion
            prediction = model.predict(roi, verbose=0)

            # Get highest probability emotion
            emotion_index = np.argmax(prediction)

            emotion = emotion_labels[emotion_index]

            # Speak emotion
            speak_emotion(emotion)

            # ================= COLORS =================

            color_map = {
                "Happy": (0, 255, 0),
                "Sad": (255, 0, 0),
                "Angry": (0, 0, 255),
                "Surprise": (255, 255, 0),
                "Neutral": (255, 255, 255),
                "Fear": (255, 0, 255),
                "Disgust": (0, 255, 255)
            }

            color = color_map.get(emotion, (0, 255, 0))

            # ================= FACE RECTANGLE =================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # ================= LABEL BACKGROUND =================

            cv2.rectangle(
                frame,
                (x, y - 40),
                (x + w, y),
                (0, 0, 0),
                -1
            )

            # ================= EMOTION TEXT =================

            cv2.putText(
                frame,
                emotion,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ================= CONVERT FRAME =================

        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=False)