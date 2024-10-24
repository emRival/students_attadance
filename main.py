import cv2
import base64
import threading
from roboflow import Roboflow
import requests
import datetime
import numpy as np
import time
from queue import Queue

# Load Roboflow model
rf = Roboflow(api_key="Yp7WftphMIr2hxMHplQ7")
project = rf.workspace().project("face-dkcci")
model = project.version(1).model

scale_percent = 25
queue = Queue()

# Initialize last_prediction_time
last_prediction_time = datetime.datetime.now()

# Function to resize the frame based on a scale factor
def resize_frame(frame, scale_percent=25):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# Function to encode image to base64
def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Function to process and send data to the API
def process_and_send_data():
    while True:
        if not queue.empty():
            student_name, status, entry_time, image_base64 = queue.get()
            data = {
                "name": student_name,
                "status": status,
                "time": entry_time,
                "image": image_base64
            }
            try:
                response = requests.post("http://localhost:3000/api/predic", json=data)
                response.raise_for_status()
                print(f"Data sent for {student_name} at {entry_time} with status {status}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to send data: {str(e)}")

# Function to perform predictions
def perform_predictions(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized_gray = resize_frame(frame_gray, scale_percent)
    
    result = model.predict(frame_resized_gray, confidence=40).json()

    return result

# Start the thread for processing and sending data
threading.Thread(target=process_and_send_data, daemon=True).start()

# Open video stream from camera
cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_display = resize_frame(frame, scale_percent)

        # Perform prediction every 3 seconds
        current_time = datetime.datetime.now()
        if (current_time - last_prediction_time).total_seconds() > 3:  # Process every 3 seconds
            last_prediction_time = current_time
            
            result = perform_predictions(frame)  # Call the prediction function

            if result['predictions']:
                for obj in result['predictions']:
                    student_name = obj['class']
                    x, y, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
                    cv2.rectangle(frame_display, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    entry_time = current_time.strftime('%H:%M')
                    school_start_time = datetime.datetime.combine(current_time.date(), datetime.time(7, 30))
                    status = "late" if current_time > school_start_time else "on time"

                    # Encode image as base64 and add to queue
                    image_base64 = encode_image_to_base64(frame)
                    queue.put((student_name, status, entry_time, image_base64))

            else:
                print("No objects detected.")

        cv2.imshow("Camera Feed", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"General error: {str(e)}")
        break

cap.release()
cv2.destroyAllWindows()
