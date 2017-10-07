//============================================================================
// Name        : FaceRecognizer.cpp
// Author      : Guru Prasad Singh
// Version     :
// Copyright   : MIT Licence
// Description : Face Recognizer uisng OpenCV 2.4 in C++, Ansi-style
//============================================================================

/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "FaceRecognizer.h"


//rad scv file and creae image and label data.
static void read_csv(const string& filename, vector<Mat>& images,
		vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message =
				"No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}



int main(int argc, const char *argv[]) {

	if (argc != 2) {
		cout << "usage: " << argv[0] << " <csv.ext>" << endl;
		exit(1);
	}

	//devie id is id of the camera, starts with zero and increments depending upon number
	//of camera attached to the system
	int deviceId = 0;

	string face_cascade_name =
			"/home/guru/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;

	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}

	// Load the cascade
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}

	string person[] = { "Dhairya", "Guru" }; //don't do this, you need to check your csv file every time you generate it

	string fn_csv = string(argv[1]);
	vector<Mat> images;
	vector<int> labels;

	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg
				<< endl;
		exit(1);
	}

	if (images.size() <= 1) {
		string error_message =
				"This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	// Holds the current frame from the Video device:
	Mat frame;
	for (;;) {
		cap >> frame;

		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector<Rect_<int> > faces;
		face_cascade.detectMultiScale(gray, faces);

		for (int i = 0; i < faces.size(); i++) {
			// Process face by face:
			Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);

			// Now perform the prediction, see how easy that is:
			int prediction = model->predict(face);
			// And finally write all we've found out to the original image!
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);
			// Create the text we will annotate the box with:

			string box_text = person[prediction];
			// Calculate the position for annotated text (make sure we don't

			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN,
					1.0, CV_RGB(0,255,0), 2.0);
		}
		// Show the result:
		imshow("face_recognizer", original);
		// And display it:
		char key = (char) waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
	return 0;
}
