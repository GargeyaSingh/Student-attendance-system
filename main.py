import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known face encodings and names
gar_image = face_recognition.load_image_file("faces/gargeya.jpg")
gars_encoding = face_recognition.face_encodings(gar_image)[0]

gupt_image = face_recognition.load_image_file("faces/guptaji.jpg")
gupts_encoding = face_recognition.face_encodings(gupt_image)[0]

known_face_encodings = [gars_encoding, gupts_encoding]
known_face_names = ["Gargeya", "Guptaji"]

# Create a dictionary to track student attendance
attendance_dict = {name: False for name in known_face_names}

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Create and open the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_name = f"{current_date}.csv"
f = open(csv_file_name, "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Date", "Time", "Status"])  # Write header

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distance)
            name = known_face_names[best_match_index]
            if not attendance_dict[name]:
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_date, current_time, "Present"])
                attendance_dict[name] = True

        # Draw a rectangle and label on the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the CSV file, release resources, and destroy windows
f.close()
video_capture.release()
cv2.destroyAllWindows()
