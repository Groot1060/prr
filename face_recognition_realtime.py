import cv2
import face_recognition
import pickle
import sqlite3
from datetime import datetime

# Load known faces
with open('known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

video_capture = cv2.VideoCapture(0)

def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, timestamp TEXT)''')
    c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, datetime.now()))
    conn.commit()
    conn.close()

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "vishal"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            mark_attendance(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
