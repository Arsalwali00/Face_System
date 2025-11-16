import cv2
import face_recognition
import pickle
import json

# Load encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Load person info
with open("persons.json", "r") as f:
    persons = json.load(f)

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)

        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = data["names"][index]

            info = persons.get(name, {"error": "No data found"})
            print(json.dumps(info, indent=4))

            # Draw rectangle & name
            color = (0, 255, 0)
        else:
            info = {"name": "Unknown"}
            print(json.dumps(info, indent=4))
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, info["name"], (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
