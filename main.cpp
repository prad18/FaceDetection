/*#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;
using namespace dnn;


int main() {
    // Load the pre-trained face detection model (Haar cascade)
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("D:/Code/C codes/FaceDetection/haarcascade_frontalface_default.xml")) {
        cerr << "Error: Could not load face detection model." << endl;
        return -1;
    }

    // Load YOLO model and configuration
    String modelConfiguration = "D:/Code/C codes/FaceDetection/yolov3.cfg";
    String modelWeights = "D:/Code/C codes/FaceDetection/yolov3.weights";
    Net yolo_net = readNetFromDarknet(modelConfiguration, modelWeights);
    yolo_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    yolo_net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video capture object
    VideoCapture video_capture;
    if (!video_capture.open(0)) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    namedWindow("Object Detection", WINDOW_NORMAL);

    while (true) {
        Mat frame;
        video_capture >> frame;

        if (frame.empty()) {
            cerr << "Error: Couldn't capture frame." << endl;
            break;
        }

        Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        // Haar cascade face detection
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.3, 5);

        // YOLO object detection
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        yolo_net.setInput(blob);

        vector<String> outNames = yolo_net.getUnconnectedOutLayersNames();
        vector<Mat> outs;
        yolo_net.forward(outs, outNames);

        // Process the YOLO detection results and draw bounding boxes
        float confidenceThreshold = 0.5;
        for (const Mat& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                Mat row = out.row(i);
                Point classIdPoint, confidencePoint, leftTop, rightBottom;
                float* p = row.ptr<float>();

                float confidence = p[4];
                if (confidence > confidenceThreshold) {
                    int classId = max_element(p + 5, p + out.cols) - p;
                    float centerX = p[0] * frame.cols;
                    float centerY = p[1] * frame.rows;
                    float width = p[2] * frame.cols;
                    float height = p[3] * frame.rows;

                    leftTop.x = centerX - width / 2;
                    leftTop.y = centerY - height / 2;
                    rightBottom.x = centerX + width / 2;
                    rightBottom.y = centerY + height / 2;

                    rectangle(frame, leftTop, rightBottom, Scalar(0, 255, 0), 2);
                    putText(frame, "Object", leftTop, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                }
            }
        }

        // Process Haar cascade face detection results
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, "Human", Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        imshow("Object Detection", frame);

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

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    // Load the cascades
    CascadeClassifier face_cascade;
    if(!face_cascade.load("D:/Code/C codes/FaceDetection/haarcascade_frontalface_default.xml")) {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    // Load the YOLO model
   String modelConfiguration = "D:/Code/C codes/FaceDetection/yolov3.cfg";
    String modelWeights = "D:/Code/C codes/FaceDetection/yolov3.weights";
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

// Open the default camera
cv::VideoCapture capture(0);
if(!capture.isOpened()) {
    printf("--(!)Error opening video capture\n");
    return -1;
}

// Set the video resolution to 640x480 for faster processing
capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

// Load the YOLO tiny model for faster processing
String modelConfiguration = "D:/Code/C codes/FaceDetection/yolov3-tiny.cfg";
String modelWeights = "D:/Code/C codes/FaceDetection/yolov3-tiny.weights";
dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

Mat frame;
int frameNumber = 0;
    while(capture.read(frame)) {
        if(frame.empty()) {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        // Resize the frame to half the original size
        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        // Process every 3rd frame
        if (frameNumber % 3 == 0) {
            // Convert the frame to grayscale
            Mat frame_gray;
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);

            // Detect faces
            vector<Rect> faces;
            face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

            // Draw rectangles around the faces and label them
            for(size_t i = 0; i < faces.size(); i++) {
                rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
                putText(frame, "Human", Point(faces[i].x, faces[i].y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }

            // Detect objects using YOLO
            Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(416, 416), Scalar(), true, false);
            net.setInput(blob);
            vector<Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());

            // Process the YOLO detections
            float confidenceThreshold = 0.6;
            for (const Mat& out : outs) {
                for (int i = 0; i < out.rows; ++i) {
                    Mat row = out.row(i);
                    Point classIdPoint, confidencePoint, leftTop, rightBottom;
                    float* p = row.ptr<float>();

                    float confidence = p[4];
                    if (confidence > confidenceThreshold) {
                        int classId = max_element(p + 5, p + out.cols) - p;
                        float centerX = p[0] * frame.cols;
                        float centerY = p[1] * frame.rows;
                        float width = p[2] * frame.cols;
                        float height = p[3] * frame.rows;

                        leftTop.x = centerX - width / 2;
                        leftTop.y = centerY - height / 2;
                        rightBottom.x = centerX + width / 2;
                        rightBottom.y = centerY + height / 2;

                        rectangle(frame, leftTop, rightBottom, Scalar(0, 255, 0), 2);
                        putText(frame, "Object", leftTop, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        // Display the frame
        imshow("Object Detection", frame);

        char key = waitKey(30);
        if (key == 'q' || key == 27) {
            break;}
        frameNumber++;}
    capture.release();
    destroyAllWindows();
    return 0;
}*/


#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    // Load the cascades
    CascadeClassifier face_cascade;
    if(!face_cascade.load("D:/Code/C codes/FaceDetection/haarcascade_frontalface_default.xml")) {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    // Load the YOLO model
    String modelConfiguration = "D:/Code/C codes/FaceDetection/yolov3.cfg";
    String modelWeights = "D:/Code/C codes/FaceDetection/yolov3.weights";
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

    // Open the default camera
    cv::VideoCapture capture(0);
    if(!capture.isOpened()) {
        printf("--(!)Error opening video capture\n");
        return -1;
    }

    // Set the video resolution to 640x480 for faster processing
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame;
    int frameNumber = 0;
    while(capture.read(frame)) {
        if(frame.empty()) {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        // Resize the frame to half the original size
        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        // Process every 3rd frame
        if (frameNumber % 3 == 0) {
            // Convert the frame to grayscale
            Mat frame_gray;
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);

            // Detect faces
            vector<Rect> faces;
            face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

            // Draw rectangles around the faces and label them
            for(size_t i = 0; i < faces.size(); i++) {
                rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
                putText(frame, "Human", Point(faces[i].x, faces[i].y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }

            // Detect objects using YOLO
            Mat blob = dnn::blobFromImage(frame, 1/255.0, Size(416, 416), Scalar(), true, false);
            net.setInput(blob);
            vector<Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());

            // Process the YOLO detections

            float confidenceThreshold = 0.6;
            vector<string> classes;
            ifstream ifs("D:/Code/C codes/FaceDetection/coco.names");
            string line;
            while (getline(ifs, line)) classes.push_back(line);
            for (const Mat& out : outs) {
                for (int i = 0; i < out.rows; ++i) {
                    Mat row = out.row(i);
                    Point classIdPoint, confidencePoint, leftTop, rightBottom;
                    float* p = row.ptr<float>();

                    float confidence = p[4];
                    if (confidence > confidenceThreshold) {
                        int classId = max_element(p + 5, p + out.cols) - p;
                        float centerX = p[0] * frame.cols;
                        float centerY = p[1] * frame.rows;
                        float width = p[2] * frame.cols;
                        float height = p[3] * frame.rows;

                        leftTop.x = centerX - width / 2;
                        leftTop.y = centerY - height / 2;
                        rightBottom.x = centerX + width / 2;
                        rightBottom.y = centerY + height / 2;

                        rectangle(frame, leftTop, rightBottom, Scalar(0, 0, 255), 2);
                        putText(frame, classes[classId], leftTop, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        // Display the frame
        imshow("Object Detection", frame);
        char key = waitKey(30);
        if (key == 'q' || key == 27) {break;}
        frameNumber++;
    }
    capture.release();
    destroyAllWindows();
    return 0;
}
