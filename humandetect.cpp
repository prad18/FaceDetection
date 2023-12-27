#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {
    // Load the pre-trained face detection model
    CascadeClassifier face_cascade;
    if (!face_cascade.load("D:/Code/C codes/FaceDetection/haarcascade_frontalface_default.xml")) {
        cerr << "Error: Could not load face detection model." << endl;
        return -1;
    }

    // Open a video capture object
    VideoCapture video_capture;
    if (!video_capture.open(0)) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    namedWindow("Face Detection", WINDOW_NORMAL);

    while (true) {
        Mat frame;
        video_capture >> frame;

        if (frame.empty()) {
            cerr << "Error: Couldn't capture frame." << endl;
            break;
        }

        Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.3, 5);

        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);

            // Display the classification result
            putText(frame, "Human", Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        imshow("Face Detection", frame);

        char key = waitKey(30);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    // Release resources
    video_capture.release();
    destroyAllWindows();

    return 0;
}
