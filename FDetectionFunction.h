#ifndef FDetectionFunction_h
#define FDetctionFunction_h

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp" 
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

inline void mouseHandler(int event, int x, int y, int flags, void* image)
{
	Mat* I = (Mat*)image;
	if (y > (*I).rows || x > (*I).cols)
		cout << "Exceed Dimension, size(I):\t" << (*I).rows << "\t" << (*I).cols << endl;
	else{
		if (event == EVENT_LBUTTONDOWN)
		{
			if ((*I).channels() == 3){
				Vec3b d = (*I).at<Vec3b>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" <<
					"BGR: " << (int)d.val[0] << ", " << (int)d.val[1] << ", " << (int)d.val[2] << endl;
			}
			else if ((*I).channels() == 1){
				Scalar e = (*I).at<uchar>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" << "Value: " << e.val[0] << endl;
			}

		}
		else if (event == EVENT_RBUTTONDOWN)
		{
			if ((*I).channels() == 3){
				Vec3b d = (*I).at<Vec3b>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" <<
					"BGR: " << (int)d.val[0] << ", " << (int)d.val[1] << ", " << (int)d.val[2] << endl;
			}
			else if ((*I).channels() == 1){
				Scalar e = (*I).at<uchar>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" << "Value: " << e.val[0] << endl;
			}
		}
		else if (event == EVENT_MBUTTONDOWN)
		{
			if ((*I).channels() == 3){
				Vec3b d = (*I).at<Vec3b>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" <<
					"BGR: " << (int)d.val[0] << ", " << (int)d.val[1] << ", " << (int)d.val[2] << endl;
			}
			else if ((*I).channels() == 1){
				Scalar e = (*I).at<uchar>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" << "Value: " << e.val[0] << endl;
			}
		}
		else if (event == EVENT_MOUSEMOVE)
		{
			if ((*I).channels() == 3){
				Vec3b d = (*I).at<Vec3b>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" <<
					"BGR: " << (int)d.val[0] << ", " << (int)d.val[1] << ", " << (int)d.val[2] << endl;
			}
			else if ((*I).channels() == 1){
				Scalar e = (*I).at<uchar>(y, x);
				cout << "Coordinate: " << x << "\t" << y << "\t" << "Value: " << e.val[0] << endl;
			}
		}
	}
};

inline void medFilt2(Mat I, Mat I_median, int kernel[]){
	int ki = kernel[0] / 2;
	int kj = kernel[1] / 2;
	int kernelSize = (kernel[0] * kernel[1]) / 2;
	int* zero = new int[I.cols];
	Scalar temp;
	for (int i = 0; i < I.cols; i++)
		zero[i] = 0;

	//Store zero count of each column upto kernel[0] rows 
	for (int i = 0; i < kernel[0]; i++){
		for (int j = 0; j < I.cols; j++){
			temp = I.at<uchar>(i, j);
			if ((int)temp.val[0] == 0)
				zero[j]++;
		}
	}

	int zeroKernel = 0;
	for (int i = 0; i < kernel[1]; i++)
		zeroKernel += zero[i];

	for (int i = ki; i < I.rows - ki - 1; i++){
		for (int j = kj; j < I.cols - kj; j++){
			if (j == kj)
				I_median.at<uchar>(i, j) = (zeroKernel <= kernelSize) ? 255 : 0;
			else{
				zeroKernel -= zero[j - kj - 1];
				zeroKernel += zero[j + kj];
				I_median.at<uchar>(i, j) = (zeroKernel <= kernelSize) ? 255 : 0;
			}
		}

		Scalar temp2;
		for (int k = 0; k < I.cols; k++){
			temp = I.at<uchar>(i - ki, k);
			temp2 = I.at<uchar>(i + ki + 1, k);
			int value1 = ((int)temp.val[0] == 0) ? 1 : 0;
			int value2 = ((int)temp2.val[0] == 0) ? 1 : 0;
			zero[k] = zero[k] + value2 - value1;
		}

		zeroKernel = 0;
		for (int k = 0; k < kernel[1]; k++)
			zeroKernel += zero[k];
	}
}

inline void bitwise_shift(Mat I){
	for (int i = 0; i < I.rows; i++)
		for (int j = 0; j<I.cols; j++)
			I.at<uchar>(i, j) = (int)(I.at<uchar>(i, j)) >> 1;
}

//Static Algorithm
inline void getHSV(Mat I, Mat I_hsv_bw, int minThresh[], int maxThresh[], int kernel[]){

  	Mat I1;// = I;
	cvtColor(I, I1, CV_BGR2HSV_FULL, 0);

	Mat hue(I.rows, I.cols, CV_8UC1);
	Mat value(I.rows, I.cols, CV_8UC1);

	int from_to[] = { 0, 0 };
	mixChannels(&I1, 1, &hue, 1, from_to, 1);
	from_to[0] = 2;
	from_to[1] = 0;
	mixChannels(&I1, 1, &value, 1, from_to, 1);

	Mat I_bw1(I.rows, I.cols, CV_8UC1);
	Mat I_bw2(I.rows, I.cols, CV_8UC1);

	threshold(hue, I_bw1, minThresh[0], 255, THRESH_BINARY);
	threshold(hue, I_bw2, maxThresh[0], 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, hue, noArray());

	threshold(value, I_bw1, minThresh[1], 255, THRESH_BINARY);
	threshold(value, I_bw2, maxThresh[1], 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, value, noArray());

	bitwise_and(hue, value, I_hsv_bw, noArray());

	medFilt2(I_hsv_bw, I_hsv_bw, kernel);
}

inline void getHist(Mat I, Mat I_hist_bw, Mat value, int thresh, int kernel[]){
	//Mat I_hist_bw(I.rows, I.cols, CV_8UC1);
	threshold(I, I_hist_bw, thresh, 255, THRESH_BINARY_INV);
	bitwise_and(I_hist_bw, value, I_hist_bw, noArray());
	medFilt2(I_hist_bw, I_hist_bw, kernel);

}

//Half Dynamic Algorithm
inline int getHSVHist(Mat I, Mat I_hsv_hist_bw, int thresh[], int pointThresh, int kernel[]){
	Mat I1;
	cvtColor(I, I1, CV_BGR2HSV_FULL, 0);
	vector<Mat> hsvPlanes;
	split(I1, hsvPlanes);
	Mat I_bw1(I.rows, I.cols, CV_8UC1);
	Mat I_bw2(I.rows, I.cols, CV_8UC1);

	threshold(hsvPlanes[0], I_bw1, thresh[0], 255, THRESH_BINARY);
	threshold(hsvPlanes[0], I_bw2, thresh[1], 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, hsvPlanes[0], noArray());

	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&hsvPlanes[2], 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
	int sum = 0;
	int maxV = 0;
	Scalar e;
	while (sum < pointThresh){
		e = (hist).at<float>(maxV);
		sum += (int)e.val[0];
		maxV += 1;
	}
	if (sum>pointThresh)
		maxV = maxV - 1;
	threshold(hsvPlanes[2], hsvPlanes[2], maxV, 255, THRESH_BINARY_INV);
	
	bitwise_and(hsvPlanes[0], hsvPlanes[2], I_hsv_hist_bw, noArray());
	medFilt2(I_hsv_hist_bw, I_hsv_hist_bw, kernel);
	return maxV;
}

//Fully Dynamic Algorithm
inline int getHSVHistV(Mat I, Mat I_hsv_hist_bw, int thresh[], int pointThresh, int kernel[]){
	//namedWindow("Hue", WINDOW_NORMAL);
	Mat I1(I.rows, I.cols, CV_8UC1);
	cvtColor(I, I1, CV_BGR2HSV_FULL, 0);
	vector<Mat> hsvPlanes;
	split(I1, hsvPlanes);
	Mat I_bw1(I.rows, I.cols, CV_8UC1);
	Mat I_bw2(I.rows, I.cols, CV_8UC1);

	threshold(hsvPlanes[0], I_bw1, thresh[0], 255, THRESH_BINARY);
	threshold(hsvPlanes[0], I_bw2, thresh[1], 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, hsvPlanes[0], noArray());
	//imshow("Hue", hsvPlanes[0]);
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&hsvPlanes[2], 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
	int sum = 0;
	int maxV = 0;
	Scalar e;
	while (sum < pointThresh){
		e = (hist).at<float>(maxV);
		sum += (int)e.val[0];
		maxV += 1;
	}
	if (sum>pointThresh)
		maxV = maxV - 1;
	cout << "PointsNum:\t" << sum << "\t" << pointThresh << endl;
	threshold(hsvPlanes[2], hsvPlanes[2], maxV, 255, THRESH_BINARY_INV);

	bitwise_and(hsvPlanes[0], hsvPlanes[2], I_hsv_hist_bw, noArray());
	medFilt2(I_hsv_hist_bw, I_hsv_hist_bw, kernel);
	return maxV;
}

inline int getHSVHistHV(Mat I, Mat I_hsv_hist_bw, int pointThresh, int kernel[]){
	Mat hist;
	Mat hist2;
	int sum = 0;
	int max = 0;
	Scalar e;
	//namedWindow("Hue", WINDOW_NORMAL);
	//namedWindow("Value", WINDOW_NORMAL);
	//namedWindow("Resultant", WINDOW_NORMAL);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat I1;
	cvtColor(I, I1, CV_BGR2HSV_FULL, 0);
	vector<Mat> hsvPlanes;
	split(I1, hsvPlanes);
	calcHist(&hsvPlanes[2], 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
	while (sum < pointThresh && max<150){
		e = (hist).at<float>(max);
		sum += (int)e.val[0];
		max += 1;
	}
	threshold(hsvPlanes[2], hsvPlanes[2], max, 255, THRESH_BINARY_INV);
	//imshow("Value", hsvPlanes[2]);
	/*
	calcHist(&hsvPlanes[0], 1, 0, hsvPlanes[2], hist2, 1, &histSize, &histRange, true, false);
	sum = 0;
	int min = 0;
	max = 0;
	while (max<40){
		e = (hist2).at<float>(max);
		sum += (int)e.val[0];
		max += 1;
	}
	int max2 = min;
	int maxSum = sum;
	while (min < 130){
		e = (hist2).at<float>(min + 40);
		sum += (int)e.val[0];
		e = (hist2).at<float>(min);
		sum -= (int)e.val[0];
		min += 1;
		if (maxSum < sum){
			max2 = min;
			maxSum = sum;
		}
	}
	max = max2 + 40;
	cout << "Max hue " << max << " " << maxSum << " " << max2 << endl;
	Mat I_bw(I.rows, I.cols, CV_8UC1);
	threshold(hsvPlanes[0], I_bw, max, 255, THRESH_BINARY_INV);
	threshold(hsvPlanes[0], hsvPlanes[0], max2, 255, THRESH_BINARY);
	bitwise_and(hsvPlanes[0], I_bw, hsvPlanes[0], noArray());
	imshow("Hue", hsvPlanes[0]);
	bitwise_and(hsvPlanes[0], hsvPlanes[2], I_hsv_hist_bw, noArray());
	*/
	medFilt2(hsvPlanes[2], I_hsv_hist_bw, kernel);
	//imshow("Resultant", I_hsv_hist_bw);
	//waitKey(0);
	return max;
}

inline void getRed(Mat I, Mat I_red_bw, Mat I_red_diff, Mat greenPlaneO, int minThresh,
	int maxThresh, int minThresh2, int maxThresh2, int kernel[]){
	Mat redPlane(I.rows, I.cols, CV_8UC1);
	Mat greenPlane(I.rows, I.cols, CV_8UC1);
	Mat bluePlane(I.rows, I.cols, CV_8UC1);

	int from_to[] = { 2, 0 };
	mixChannels(&I, 1, &redPlane, 1, from_to, 1);
	from_to[0] = 1;
	from_to[1] = 0;
	mixChannels(&I, 1, &greenPlane, 1, from_to, 1);
	greenPlane.copyTo(greenPlaneO);
	from_to[0] = 0;
	from_to[1] = 0;
	mixChannels(&I, 1, &bluePlane, 1, from_to, 1);

	Mat I_bw1(I.rows, I.cols, CV_8UC1);
	Mat I_bw2(I.rows, I.cols, CV_8UC1);

	threshold(greenPlane, I_bw1, minThresh2, 255, THRESH_BINARY);
	threshold(greenPlane, I_bw2, maxThresh2, 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, I_bw1, noArray());

	threshold(redPlane, redPlane, 255 / 2, 255, THRESH_TOZERO_INV);

	//greenPlane = 0.5*greenPlane;
	//bluePlane = 0.5*bluePlane;
	bitwise_shift(greenPlane);
	bitwise_shift(bluePlane);
	subtract(redPlane, greenPlane, I_red_diff, noArray(), -1);
	subtract(I_red_diff, bluePlane, I_red_diff, noArray(), -1);
	Mat I_bw_temp(I.rows, I.cols, CV_8UC1);
	threshold(I_red_diff, I_red_bw, minThresh, 255, THRESH_BINARY);
	threshold(I_red_diff, I_bw_temp, maxThresh, 255, THRESH_BINARY_INV);
	bitwise_and(I_red_bw, I_bw_temp, I_red_bw, noArray());
	bitwise_and(I_bw1, I_red_bw, I_red_bw, noArray());
	medFilt2(I_red_bw, I_red_bw, kernel);
};

inline void constrain(int min, int max, int* value){
	if ((*value) < min)
		*value = min;
	if ((*value) > max)
		*value = max;
};

struct Blob {
	vector<double> contourareas;
	vector<Point> centroids;
	vector<double> eccentricity;
	vector<vector<double> > dimension;
	vector<vector<Point> > contours;
	vector<float> angle;
};

inline void getBlobProps(Mat I, Blob* myBlobs)
{
	vector<Vec4i> hierarchy;
	findContours(I, myBlobs->contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i<(myBlobs->contours).size(); i++) {
		vector<Point> contour = (myBlobs->contours)[i];
		(myBlobs->contourareas).push_back(contourArea(contour));
		Moments M = moments(contour);
		(myBlobs->centroids).push_back(Point(M.m10 / M.m00, M.m01 / M.m00));
		(myBlobs->dimension).push_back(*(new vector<double>));

		if ((myBlobs->contours)[i].size() > 5){
			RotatedRect ell = fitEllipse(Mat((myBlobs->contours)[i]));
			double a = ell.size.width / 2;    // width >= height
			double b = ell.size.height / 2;
			(myBlobs->dimension[(myBlobs->dimension).size() - 1]).push_back(ell.size.width);
			(myBlobs->dimension[(myBlobs->dimension).size() - 1]).push_back(ell.size.height);
			if (a > b)
				(myBlobs->eccentricity).push_back(0);
			else
				(myBlobs->eccentricity).push_back(sqrt(1 - (a*a) / (b*b)));
			(myBlobs->angle).push_back(ell.angle);
		}
		else{
			RotatedRect rect = minAreaRect(Mat(contour));
			(myBlobs->dimension[i]).push_back(rect.size.width);
			(myBlobs->dimension[i]).push_back(rect.size.height);
			//(myBlobs->eccentricity).push_back(sqrt(1 - (rect.size.width*rect.size.width) / (rect.size.height*rect.size.height)));
			(myBlobs->eccentricity).push_back(0);
			(myBlobs->angle).push_back(rect.angle);
		}
	}
}

inline double CalcMedian(vector<int> scores)
{
	double median;
	size_t size = scores.size();

	if (size == 0)
		return 0;

	sort(scores.begin(), scores.end());

	if (size % 2 == 0)
	{
		median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
	}
	else
	{
		median = scores[size / 2];
	}

	return median;
}

//Pole Detection
void getWhiteHSV(Mat I, Mat I_gray_bw, int minThresh[], int maxThresh[], int kernel[]){
	cvtColor(I, I, CV_BGR2HSV_FULL, 0);
	Mat hue(I.rows, I.cols, CV_8UC1);
	Mat saturation(I.rows, I.cols, CV_8UC1);
	Mat value(I.rows, I.cols, CV_8UC1);

	int from_to[] = { 0, 0 };
	mixChannels(&I, 1, &hue, 1, from_to, 1);
	from_to[0] = 1;
	from_to[1] = 0;
	mixChannels(&I, 1, &saturation, 1, from_to, 1);
	from_to[0] = 2;
	from_to[1] = 0;
	mixChannels(&I, 1, &value, 1, from_to, 1);

	Mat I_bw_hue(I.rows, I.cols, CV_8UC1);
	Mat I_bw_saturation(I.rows, I.cols, CV_8UC1);
	Mat I_bw_value(I.rows, I.cols, CV_8UC1);

	threshold(hue, I_bw_hue, minThresh[0], 255, CV_THRESH_BINARY);
	threshold(hue, hue, maxThresh[0], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_hue, hue, I_bw_hue, noArray());

	threshold(saturation, I_bw_saturation, minThresh[1], 255, CV_THRESH_BINARY);
	threshold(saturation, saturation, maxThresh[1], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_saturation, saturation, I_bw_saturation, noArray());

	threshold(value, I_bw_value, minThresh[2], 255, CV_THRESH_BINARY);
	threshold(value, value, maxThresh[2], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_value, value, I_bw_value, noArray());

	I_gray_bw = I_bw_hue & I_bw_saturation & I_bw_value;
	medFilt2(I_gray_bw, I_gray_bw, kernel);
};

void getWhiteRGB(Mat I, Mat I_gray_bw, int minThresh[], int maxThresh[], int diffThresh, int kernel[]){

	
	Mat redPlane(I.rows, I.cols, CV_8UC1);
	Mat greenPlane(I.rows, I.cols, CV_8UC1);
	Mat bluePlane(I.rows, I.cols, CV_8UC1);

	int from_to[] = { 2, 0 };
	mixChannels(&I, 1, &redPlane, 1, from_to, 1);
	from_to[0] = 1;
	from_to[1] = 0;
	mixChannels(&I, 1, &greenPlane, 1, from_to, 1);
	from_to[0] = 0;
	from_to[1] = 0;
	mixChannels(&I, 1, &bluePlane, 1, from_to, 1);

	Mat I_bw_blue(I.rows, I.cols, CV_8UC1);
	Mat I_bw_green(I.rows, I.cols, CV_8UC1);
	Mat I_bw_red(I.rows, I.cols, CV_8UC1);

	threshold(bluePlane, I_bw_blue, minThresh[0], 255, CV_THRESH_BINARY);
	threshold(bluePlane, bluePlane, maxThresh[0], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_blue, bluePlane, I_bw_blue, noArray());

	threshold(greenPlane, I_bw_green, minThresh[1], 255, CV_THRESH_BINARY);
	threshold(greenPlane, greenPlane, maxThresh[1], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_green, greenPlane, I_bw_green, noArray());

	threshold(redPlane, I_bw_red, minThresh[2], 255, CV_THRESH_BINARY);
	threshold(redPlane, redPlane, maxThresh[2], 255, CV_THRESH_BINARY_INV);
	bitwise_and(I_bw_red, redPlane, I_bw_red, noArray());
	
	subtract(greenPlane, redPlane, greenPlane, noArray());
	subtract(redPlane, bluePlane, redPlane, noArray());
	subtract(redPlane, greenPlane, redPlane, noArray());
	threshold(redPlane, redPlane, diffThresh, 255, CV_THRESH_BINARY_INV);

	I_gray_bw = I_bw_blue & I_bw_green & I_bw_red&redPlane;
	medFilt2(I_gray_bw, I_gray_bw, kernel);
};


struct Rectangle{
	vector<Point> centroids;
	vector<vector<Point> > contours;
	vector<vector<int> > dimension;
};

void getRectProps(Mat I, Rectangle* myRectangle)
{
	vector<Vec4i> hierarchy;
	findContours(I, myRectangle->contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i<(myRectangle->contours).size(); i++) {
		vector<Point> contour = (myRectangle->contours)[i];
		RotatedRect rect = minAreaRect(Mat((myRectangle->contours)[i]));
		(myRectangle->dimension).push_back(*(new vector<int>));
		(myRectangle->dimension)[i].push_back(rect.size.width);
		(myRectangle->dimension)[i].push_back(rect.size.height);
		(myRectangle->dimension)[i].push_back(rect.angle);
		(myRectangle->centroids).push_back(rect.center);
	}
}
#endif
