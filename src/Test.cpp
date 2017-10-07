#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void cropImage(Mat frame);


string face_cascade_name =		"/home/guru/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";  //change it to your xml location
CascadeClassifier face_cascade;

int filenumber;
string filename;

// Function main
int main1(void) {
	// Load the cascade
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}

	cv::String path("/home/guru/OpenCV WorkSpace/FaceRecognizer/*.jpg"); //select only jpg
	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(path, fn, true); // recurse
	for (size_t k = 0; k < fn.size(); ++k) {

		Mat frame = cv::imread(fn[k]);

		// Read the image file
		// Mat frame = imread("/home/guru/OpenCV WorkSpace/FaceRecognizer/Debug/1.jpg");

		// Apply the classifier to the frame
		if (!frame.empty()) {
			cropImage(frame);
		} else {
			printf(" --(!) No captured frame -- Break!");
			//break;
		}

		int c = waitKey(10);

		if (27 == char(c)) {
			//break;
		}
	}

	return 0;
}

// Function detectAndDisplay
void cropImage(Mat frame) {
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0;
	int ac = 0;

	size_t ib = 0;
	int ab = 0;

	for (ic = 0; ic < faces.size(); ic++)

			{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height;

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height;

		if (ac > ab) {
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);

		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

		// Form a filename
		filename = "";
		stringstream ssfn;
		ssfn << filenumber << ".jpg";
		filename = ssfn.str();
		filenumber++;

		imwrite(filename, gray);

	}

}
