#include "opencv2/opencv.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>

#include <windows.h> 
#include <vector>
#include <stdio.h>
#include <time.h>


using namespace cv;
using namespace std;

int main(int argc, const char** argv) {


	// Create a VideoCapture object and open the input file
	VideoCapture cap("doll.avi");
	// Check if video file opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video file" << endl;
		return -1;
	}
	// read frame rate of the video file
	double fps = cap.get(CV_CAP_PROP_FPS);
	cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;

	Mat frame, hsv, mask, hue, hist, backprojection;
	//Mat hsv_roi_ref;
	Mat bgr[3];
	double similarity_cd;
	int vmin = 10, vmax = 256, smin = 30;

	// define region of interest
	Rect roi;
	cap >> frame;

	// select region of interest
	//roi = selectROI("ROI", frame);
	/*
	// good result for car
	roi.x = 480;
	roi.y = 283;
	roi.width = 33;
	roi.height = 14;
	*/
	
	/*
	// new basket1
	roi.x = 113;
	roi.y = 340;
	roi.width = 107;
	roi.height = 140;
	*/

	/*
	// new basket2
	roi.x = 271;
	roi.y = 217;
	roi.width = 51;
	roi.height = 79;
	*/

	

	
	// doll
	roi.x = 248;
	roi.y = 294;
	roi.width = 29;
	roi.height = 64;
	


	/*
	// lemming
	roi.x = 51;
	roi.y = 233;
	roi.width = 43;
	roi.height = 71;
	*/
	

	/*
	// bike
	roi.x = 172;
	roi.y = 188;
	roi.width = 27;
	roi.height = 98;
	*/
	
	/*
	// man
	roi.x = 505;
	roi.y = 130;
	roi.width = 35;
	roi.height = 77;
	*/
	/*
	// basket1
	roi.x = 293;
	roi.y = 315;
	roi.width = 57;
	roi.height = 92;
	*/

	cout << roi.x << "\t" << roi.y << "\t" << roi.width << "\t" << roi.height << endl;



	cvtColor(frame, hsv, COLOR_BGR2HSV);          // convert from RGB to HSV

	// seperate hue channel from hsv image
	inRange(hsv, Scalar(0, smin, vmin),
		Scalar(180, 256, vmax), mask);
	int ch[] = { 0, 0 };
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);


	// calculate Histogram of the target
	Mat roi_Hue;
	Mat roi_Hue1(hue, roi), maskroi(mask, roi);
	roi_Hue1.copyTo(roi_Hue);
	int hsize = 180;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	calcHist(&roi_Hue, 1, 0, Mat() , hist, 1, &hsize, &phranges);
	normalize(hist, hist, 0, 255, NORM_MINMAX);


	// calculate saturation histogram of the target
	Mat sat_hist;
	Mat new_hsv[3];
	split(hsv, new_hsv);
	Mat satu_roi1(new_hsv[1], roi);
	Mat satu_roi;
	satu_roi1.copyTo(satu_roi);
	int ssize = 256;
	float sranges[] = { 0,255 };
	const float* psranges = sranges;
	calcHist(&satu_roi, 1, 0, Mat(), sat_hist, 1, &ssize, &psranges);
	normalize(sat_hist, sat_hist, 0, 255, NORM_MINMAX);

	
	Mat hue_roi_candidate;
	MatND hist_candidate;

	Mat sat_back;
	namedWindow("frame", 0);
	int m = 0;
	//clock_t tStart = clock();
	int d = 1;
	double processing_time = 0;
	double sum = 0;
	double average = 0;
	while (d < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
		d++;
		//if (!cap.grab())
			//break;
		clock_t tStart = clock();
		cap >> frame;
		// extract hue channel
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		// calculate back projection image
		calcBackProject(&hue, 1, 0, hist, backprojection, &phranges);
		
		m++;
		if (m == 5)
		{
			m = 0;
			// calculate back projection from hue and saturation
			split(hsv, new_hsv);
			calcBackProject(&new_hsv[1], 1, 0, sat_hist, sat_back, &psranges);
			backprojection &= sat_back;
			// implement camshift algorithm
			RotatedRect trackBox = CamShift(backprojection, roi,
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
			
			// calculate similarity between roi and traget
			hue_roi_candidate = hue(roi);
			resize(hue_roi_candidate, hue_roi_candidate, roi_Hue.size());
			calcHist(&hue_roi_candidate, 1, 0, Mat(), hist_candidate, 1, &hsize, &phranges);
			normalize(hue_roi_candidate, hue_roi_candidate, 0, 255, NORM_MINMAX);
			similarity_cd = compareHist(hist, hist_candidate, CV_COMP_CORREL);

			// update histogram  similarity_cd > 0.8
			if (similarity_cd > 0.8)
			{
				cout << "histogram updated" << endl;
				
				// create mask and multiply it by back projection image
				Mat My_mask(backprojection.size(), backprojection.type(), Scalar::all(0));
				My_mask(roi).setTo(Scalar::all(1));
				backprojection &= My_mask;
				
				//convert back projection to binary image
				threshold(backprojection, backprojection, 0, 255, THRESH_BINARY);
				int morph_size = 2;
				Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
				// close operation
				morphologyEx(backprojection, backprojection, MORPH_CLOSE, element);
				// dilate operation
				dilate(backprojection, backprojection, element);
				// increase structuring element size
				morph_size = 3;
				Mat element_2 = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
				// open operation
				morphologyEx(backprojection, backprojection, MORPH_OPEN, element_2);

				// delet all samll objects
				vector<vector<Point> > vtContours;
				vector<Vec4i> vtHierarchy;
				double dMaxArea = 200;
				int nSavedContour;
				findContours(backprojection, vtContours, vtHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

				for (int i = 0; i < vtContours.size(); i++)
				{
					double area = contourArea(vtContours[i]);

					if (area <= 100) {
						continue;
					}

					// Get only one
					if (area > dMaxArea)
					{
						dMaxArea = area;
						nSavedContour = i;
					}

				}

				if (nSavedContour == -1)
				{
					return false;
				}

				backprojection = Scalar::all(0);
				drawContours(backprojection, vtContours, nSavedContour, Scalar(255), CV_FILLED, 8);

				// implement camshift algorithm
				RotatedRect trackBox = CamShift(backprojection, roi,
					TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));


				// update hue histogram
				Mat new_roi_Hue;
				Mat roi_updated(hue, roi);
				roi_updated.copyTo(new_roi_Hue);

				calcHist(&new_roi_Hue, 1, 0, Mat(), hist, 1, &hsize, &phranges);
				//hist.at<float>(0) = 0;
				normalize(hist, hist, 0, 255, NORM_MINMAX);
				
			}
			
			ellipse(frame, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
			imshow("frame", frame);
			
		}
		

		else
		{
			RotatedRect trackBox = CamShift(backprojection, roi,
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

			if (roi.area() <= 1)
			{
				int cols = backprojection.cols, rows = backprojection.rows, r = (MIN(cols, rows) + 5) / 6;
				roi = Rect(roi.x - r, roi.y - r,
					roi.x + r, roi.y + r) &
					Rect(0, 0, cols, rows);
			}


			// draw ellipse around the target
			ellipse(frame, trackBox, Scalar(0, 0, 255), 3, LINE_AA);

			imshow("frame", frame);
		}

	
		processing_time = (double)(clock() - tStart) / CLOCKS_PER_SEC;
		sum = sum + processing_time;
		//printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		//Sleep(500);
		char c = (char)waitKey(10);
	
	}
	average = sum / d;
	cout << average << endl;
	//printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	cout << d << "  " << cap.get(CV_CAP_PROP_FRAME_COUNT) << "   ";
	waitKey(0);

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();

waitKey(0);

	return 0;
}



