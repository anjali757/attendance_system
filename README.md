**Face Recognition Attendance System**

An AI-based attendance system that uses face recognition to automatically identify people from video or webcam and record their attendance with time, date, and confidence score.


## Features

* Face detection and recognition using **InsightFace**
* Image-based face enrollment
* Automatic IN/OUT attendance marking
* CSV attendance file generation
* Output video with face bounding boxes and names


## Technologies Used

* Python
* OpenCV
* InsightFace (Buffalo-L)
* NumPy, SciPy


## How It Works

1. Faces are enrolled from images stored in the `enroll/` folder.
2. InsightFace extracts facial embeddings and saves them in a `face_db.pkl` file.
3. Faces from video or webcam are recognized using cosine similarity.
4. Attendance is marked automatically while avoiding duplicate entries.


## Working Example
1. Add face images in the enroll folder (one folder per person).
2. Run the program to create the face database.
3. Provide video or webcam input.
4. Attendance is recorded automatically in CSV format.
 

##  Output

* **attendance.csv** – attendance records with date, time, and confidence
* **output_attendance.mp4** – processed video with detected faces and labels



