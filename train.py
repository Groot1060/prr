import face_recognition
import os
import pickle

def train_model(data_dir='dataset'):
    known_face_encodings = []
    known_face_names = []

    for kishor in os.listdir(data_dir):
        if kishor.endswith('.jpg'):
            image_path = os.path.join(data_dir, kishor)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(image_name)[0])
    
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

if __name__ == '__main__':
    train_model()
