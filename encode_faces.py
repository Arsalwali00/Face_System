import os
import face_recognition
import pickle

data_path = "faces"
encoding_file = "encodings.pickle"

known_encodings = []
known_names = []

for person in os.listdir(data_path):
    person_folder = os.path.join(data_path, person)
    
    if not os.path.isdir(person_folder):
        continue
    
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("Training completed. Encodings saved!")
