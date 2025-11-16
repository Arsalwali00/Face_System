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

# Set lower camera resolution for faster capture
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

# Performance optimizations
frame_count = 0
process_every_n_frames = 5  # Increased from 3 to 5 - process less often
scale_factor = 0.25  # Reduced from 0.5 to 0.25 - much smaller frames

# Cache last detection results
cached_faces = []
last_printed_name = None  # Track last printed person

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Only process every Nth frame
    if frame_count % process_every_n_frames == 0:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces with faster model
        boxes = face_recognition.face_locations(rgb, model="hog")  # "hog" is faster than "cnn"
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Clear cache and update with new detections
        cached_faces = []
        
        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            # Scale coordinates back to original size
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)
            
            # Use face_distance for faster matching
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = face_distances.argmin() if len(face_distances) > 0 else -1

            name = "Unknown"
            color = (0, 0, 255)
            info = {"name": "Unknown"}

            if best_match_index != -1 and face_distances[best_match_index] < 0.5:
                name = data["names"][best_match_index]
                info = persons.get(name, {"name": name, "error": "No data found"})
                color = (0, 255, 0)
                
                # Print only when a different person is detected
                if name != last_printed_name:
                    print(json.dumps(info, indent=4))
                    last_printed_name = name
            
            cached_faces.append({
                "box": (top, right, bottom, left),
                "info": info,
                "color": color
            })
    
    # Draw cached faces on every frame (for smooth video)
    for face in cached_faces:
        top, right, bottom, left = face["box"]
        color = face["color"]
        info = face["info"]
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, info["name"], (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()