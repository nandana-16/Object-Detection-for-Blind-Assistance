import cv2 
from ultralytics import YOLO 
import pyttsx3 
import threading


model = YOLO("yolov8n.pt")  
engine = pyttsx3.init()


def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()  
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
cap.set(cv2.CAP_PROP_FPS, 30)  


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

   
    results = model(frame)

    
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id] 
            confidence = box.conf.item()
            if confidence > 0.8:  
                detected_objects.add(label)

  
    if detected_objects:
        announcement = ", ".join(detected_objects)  
        print(announcement)  
        threading.Thread(target=speak, args=(announcement,)).start()  

    annotated_frame = results[0].plot() 
    cv2.imshow("Object Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()