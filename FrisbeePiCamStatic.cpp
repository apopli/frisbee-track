#include <wiringPi.h>
#include <wiringSerial.h>
#include "FDetectionFunction.h"
#include <cmath>
#include <ctime>
#include <time.h>
#include <raspicam/raspicam_cv.h>
#include <omp.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <exception>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace cv;
using namespace raspicam;

void resetVariables(Point* centroid, int* frisbeeArea, vector<int>* frisbeeX, vector<int>* frisbeeY, vector<int>* changeX,
	vector<int>* changeY, bool* windowWidthChange, bool* windowHeightChange, int* spotWidthI, int* spotWidthII, int* spotHeightI,
	int* spotHeightII, bool* decreaseArea, bool* goingUp, int* methodUsed, int* lastFrame, int* startCapture, int* nCount, int* widthBy2, int* heightBy2, int* prevWidthBy2, int* prevHeightBy2)
{
	(*centroid).x = 450;
	(*centroid).y = 300;
	*frisbeeArea = 2600;
	vector<int>().swap(*frisbeeX);
	vector<int>().swap(*frisbeeY);
	vector<int>().swap(*changeX);
	vector<int>().swap(*changeY);
	*windowWidthChange = false;
	*windowHeightChange = false;
	*spotWidthI = 25;
	*spotWidthII = 30;
	*spotHeightI = 40;
	*spotHeightII = 40;
	*decreaseArea = false;
	*goingUp = true;
	*methodUsed = 1;
	*lastFrame = *nCount;
	*startCapture = 0;
	*widthBy2 = 200;
	*heightBy2 = 200;
	*prevWidthBy2 = 400;
	*prevHeightBy2 = 400;
}

void restrictCentroid(Point* centroid, int* prevWidthBy2, int *prevHeightBy2, int *widthBy2, int *heightBy2, bool *windowWidthChange, bool *windowHeightChange, int resolution[])
{

	if (((*centroid).x - *widthBy2) < 1) {
		*prevWidthBy2 = *widthBy2;
		*widthBy2 = (*centroid).x - 1;
		*windowWidthChange = true;
	}

	if (((*centroid).x + *widthBy2) > resolution[1]) {
		*prevWidthBy2 = *widthBy2;
		*widthBy2 = resolution[1] - (*centroid).x;
		*windowWidthChange = true;
	}

	if (((*centroid).y - *heightBy2) < 1) {
		*prevHeightBy2 = *heightBy2;
		*heightBy2 = (*centroid).y - 1;
		*windowHeightChange = true;
	}

	if (((*centroid).y + *heightBy2) > resolution[0]) {
		*prevHeightBy2 = *heightBy2;
		*heightBy2 = resolution[0] - (*centroid).y;
		*windowHeightChange = true;
	}
}

int main(int argc, const char** argv) {

	int z;
	ofstream outputFile("drift.txt");
	int fd = serialOpen("/dev/ttyAMA0", 57600);
	
	if (fd == -1){
		cout << strerror(errno) << endl;
		return 0;
	}
	
	cout << "start" << endl;
	int v = 2;
	int widthBy2 = 200;
	int heightBy2 = 200;
	int prevWidthBy2 = 400;
	int prevHeightBy2 = 400;
	int kernel[] = { 17, 5 };
	int kernelWhiteI[] = {40, 20};
	int kernelWhiteII[] = { 40, 20 };
	Point centroid = Point(450, 300);

	vector<int> frisbeeX;
	vector<int> frisbeeY;
	vector<int> changeX;
	vector<int> changeY;

	int resolution[] = { 1280, 720 };
	bool windowWidthChange = false;
	bool windowHeightChange = false;

	int minThreshColor = 25;	//Min thresh for diff in RGB
	int maxThreshColor = 42;	//Max thresh for diff in RGB
	int minThreshGreen = 50;	//Min thresh for green in RGB
	int maxThreshGreen = 90;	//Max thresh for green in RGB
	int defaultMinThresh = 25;
	int defaultMaxThresh = 42;
	int defaultMinThreshGreen = 50;
	int defaultMaxThreshGreen = 90;

	//Current Color Threshold in Use
	//int minThreshHSV[] = { 20, 65 };
	//int maxThreshHSV[] = { 40, 110 };
	int minThreshHSV[] = { 80, 25 };
	int maxThreshHSV[] = { 160, 70 };
	
	// int minThreshWhiteRGBI[] = {79, 120, 105};
	// int maxThreshWhiteRGBI[] = {95, 130, 115};
	// int minThreshWhiteRGBII[] = { 95, 130, 115 };
	// int maxThreshWhiteRGBII[] = { 115, 155, 130 };
	
	int minThreshWhiteRGBI[] = {55, 60, 90};
	int maxThreshWhiteRGBI[] = {70, 80, 125};
	int minThreshWhiteRGBII[] = { 55, 65, 135 };
	int maxThreshWhiteRGBII[] = { 70, 80, 150 };
	

	int diffThreshWhiteRGBI = 25;
	int diffThreshWhiteRGBII = 10;

	//Frisbee Specification
	double eccentricityThresh = 0.8;
	int frisbeeArea = 2600;

	//Spot Specification
	Point spotCentroidI = Point(232, 730);
	Point spotCentroidII = Point(272, 625);
	int poleUsed = 2;

	int minXSpot = 50;
	int maxXSpot = 400;
	int spotWidthI = 25;
	int spotWidthII = 30;
	int spotHeightI = 40;
	int spotHeightII = 40;
  
	bool goingUp = true;
	int methodUsed = 1;
	bool decreaseArea = false;
	bool useRGB = false;
	bool useWhiteRGB = true;
  	bool frisbeeLand = false;

	time_t timer_begin, timer_end;
	long int start_time;
	long int time_difference;
	struct timespec gettime_now;
	RaspiCam_Cv Camera;
	int nCount = 50;
	int lastFrame = nCount;

	Mat image[100];
	Mat image_[100];
	int startCapture = 0;

	//set camera params
	Camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	Camera.set(CV_CAP_PROP_FORMAT, CV_8UC3);
	Camera.set(CV_CAP_PROP_BRIGHTNESS, 60);
	Camera.set(CV_CAP_PROP_SATURATION, 60);
	Camera.set(CV_CAP_PROP_CONTRAST, 60);
	//Open camera
	cout << "Opening Camera..." << endl;
	if (!Camera.open()) { cerr << "Error opening the camera" << endl; return -1; }

	cout << "---COUNTDOWN---" << endl;
	for (int i = 5; i > 0; i--) {
		cout << "-------" << i << "-------" << endl;
		sleep(1);
	}
	cout << "-----GO!-----" << endl;

	
	serialFlush(fd);
	while (!serialDataAvail(fd));
	if (serialDataAvail(fd)) {
		z = serialGetchar(fd);
		if (z == 5 || z == 6) {
			poleUsed = z-4;
			serialFlush(fd);
			goto START;
		}
	}
	outputFile << "GO"<<endl;
	//int h = 2;
START:
	//Start capture
	cout << "Capturing " << nCount << " frames ...." <<"video no. "<<v<< endl;
	time(&timer_begin);
	//stringstream convert2;
	//convert2<<h;
	//h++;
	//string name = "video"+convert2.str()+".avi";
	//VideoCapture cap("video2.avi");
	//nCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	//cout<<"Frame Count:\t"<<nCount<<endl;
	clock_gettime(CLOCK_REALTIME, &gettime_now);
	start_time = gettime_now.tv_nsec;
	cout<<poleUsed<<endl;
	outputFile<<poleUsed<<endl;
	int captured = 0;
	int k = 15;
	bool frisbeeFound = false;
	int minimum;
	int id = 0;	
	bool poleDetect = false;
	int tid;
	double frisbeeWidth = 40;
	double frisbeeHeight = 180;
	bool restart = false;
	bool waitForStart = false;
	int counter = 0;

	omp_set_nested(1);
	omp_set_num_threads(2);
#pragma omp parallel shared(id,minimum,frisbeeFound,k,captured,startCapture,widthBy2,heightBy2,prevHeightBy2,prevWidthBy2,centroid,resolution,frisbeeArea,frisbeeX,frisbeeY, Camera, changeX,changeY,windowHeightChange,windowWidthChange,minThreshColor,maxThreshColor,minThreshGreen,maxThreshGreen,defaultMaxThresh,defaultMaxThreshGreen,defaultMinThresh,defaultMinThreshGreen,methodUsed,decreaseArea,timer_begin,timer_end,start_time,time_difference,gettime_now,nCount,image,image_) private(tid)
	{
		tid = omp_get_thread_num();

		if (tid == 0) {
			Camera.grab();
			Camera.retrieve(image_[0]);
			transpose(image_[0], image_[0]);
			flip(image_[0], image_[0], 1);

			for (int t = 1; t < k; t++) {
				//cap >> image_[t];
				if (serialDataAvail(fd)) {
					z = serialGetchar(fd);
					if (z == 5 || z == 6) {
						poleUsed = z-4;
						serialFlush(fd);
						restart = true;
						break;
					} 
				}
				Camera.grab();
				Camera.retrieve(image_[t]);
			}

			//cap >> image_[k];
			Camera.grab();
			Camera.retrieve(image_[k]);
#pragma omp atomic
			startCapture++;

			for (int t = k+1; t < nCount; t++) {
				//cap >> image_[t];
				if (restart)
					break;
				if (serialDataAvail(fd)) {
					z = serialGetchar(fd);
					if (z == 5 || z == 6) {
						poleUsed = z-4;
						serialFlush(fd);
						restart = true;
					}
				}
				Camera.grab();
				Camera.retrieve(image_[t]);
			}

			//Camera.release();
#pragma omp atomic
			captured++;
		}
		else if (tid == 1) {
			omp_set_num_threads(3);

			while (!frisbeeFound){
				if (restart || captured || k>=32)
					break;
				while (image_[k].rows == 0)
				while (!startCapture);
				transpose(image_[k], image_[k]);
				flip(image_[k], image_[k], 1);

				restrictCentroid(&centroid, &prevWidthBy2, &prevHeightBy2, &widthBy2, &heightBy2, &windowWidthChange, &windowHeightChange, resolution);
				Rect region_of_interest = Rect((centroid.x - widthBy2), (centroid.y - heightBy2), 2 * widthBy2, 2 * heightBy2);
				
				try{
					image[k] = image_[k](region_of_interest);
				}
				catch(exception& e){
					cout<<"Exception While 1:\t"<<e.what()<<"\t"<<k<<endl;
					cout<<(centroid.x - widthBy2)<<"\t"<<(centroid.y - heightBy2)<<"\t"<<2 * widthBy2<<"\t"<<2 * heightBy2<<endl;
					cout << region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
					cout<<image_[k].rows<<" "<<image_[k].cols<<" "<<image_[k].channels()<<endl;
					outputFile <<"ROI in while 1" <<region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
					if (!restart)
						waitForStart = true;
					break;
				}
				Mat frame_red_bw(image[k].rows, image[k].cols, CV_8UC1);
				Mat frame_red_diff(image[k].rows, image[k].cols, CV_8UC1);
				Mat frame_greenPlane(image[k].rows, image[k].cols, CV_8UC1);

				if (useRGB)
					getRed(image[k], frame_red_bw, frame_red_diff, frame_greenPlane, minThreshColor, maxThreshColor, minThreshGreen, maxThreshGreen, kernel);
				else{
					getHSV(image[k], frame_red_bw, minThreshHSV, maxThreshHSV, kernel);
					//int thresh[] = {0, 90};
					//int maxV = getHSVHist(image[k], frame_red_bw, thresh, 4000, kernel);
					//getHSVHistHV(image[k], frame_red_bw, 4000, kernel);
					//cout << "MaxV:\t"<<maxV<<endl;
				}

				Blob* windowBlobs = new Blob;
				getBlobProps(frame_red_bw, windowBlobs);
				int numberBlobs = (windowBlobs->centroids).size();

				if (numberBlobs >= 1){
					minimum = 0;
					for (int j = 1; j < numberBlobs; j++) {
						if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
							&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
						((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)) && abs((windowBlobs->dimension)[j][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[j][0] - frisbeeWidth)<20)
							minimum = j;
					}
					//cout<<"Blobs Detected!\nArea: ";
					//for (int j=0; j<numberBlobs; j++){
					//	cout<<(windowBlobs->contourareas)[j]<<" ";
					//}
					//cout<<endl;
					if (abs(frisbeeArea - (windowBlobs->contourareas)[minimum]) < 1000 && (windowBlobs->eccentricity)[minimum] > eccentricityThresh && 
						(((windowBlobs->angle)[minimum]>340 || (windowBlobs->angle)[minimum]<20) || ((windowBlobs->angle)[minimum]>160 && (windowBlobs->angle)[minimum]<200))
						&& abs((windowBlobs->dimension)[minimum][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[minimum][0] - frisbeeWidth)<20){
						frisbeeX.push_back(centroid.x + (windowBlobs->centroids)[minimum].x - widthBy2);
						frisbeeY.push_back(centroid.y + (windowBlobs->centroids)[minimum].y - heightBy2);
						centroid.x = frisbeeX[0];
						centroid.y = frisbeeY[0];
						frisbeeArea = (windowBlobs->contourareas)[minimum];
						frisbeeFound = true;
						cout << "Found on iteration " << k << " Area = " << frisbeeArea << endl;
						outputFile << "Found on iteration " << k << " Area = " << frisbeeArea << endl;
						circle(image_[k], centroid, 5, 0, -1, 8, 0);
						imwrite("frisbee0.jpg", image_[k]);
					}
				}
				image[k].release();
				k++;
			}

			if (captured == 0) {
				widthBy2 = 200;
				prevWidthBy2 = 200;
				heightBy2 = 200;
				prevHeightBy2 = 200;
			}

			for (int i = 1; i < nCount - k + 1; i++) {
				
				if (i > 2 && restart)
					restart = false;

				id = i;
				if (waitForStart)
					break;
				if (captured == 1 || restart) {
					if (!frisbeeFound)
						id = 0;
					break;
				}

				if (i > 2 && centroid.y<300 && counter < 3){
					if (counter == 0)
						cout << "Threshold height reaches" << endl;
					counter += 1;
					frisbeeX.push_back(centroid.x + changeX[changeX.size() - 1]);
					frisbeeY.push_back(centroid.y);
					changeX.push_back(frisbeeX[frisbeeX.size()-1] - centroid.x);
					changeY.push_back(frisbeeY[frisbeeX.size()-1] - centroid.y);
					centroid.x = frisbeeX[frisbeeX.size() - 1];
					centroid.y = frisbeeY[frisbeeY.size() - 1];
					frisbeeArea = 500;
					frisbeeWidth -= 3 * 12;
					frisbeeHeight -= 3 * 1.5;
					continue;
				}

				while (image_[i + k - 1].empty());
				transpose(image_[i + k - 1], image_[i + k - 1]);
				flip(image_[i + k - 1], image_[i + k - 1], 1);

				if (centroid.x < 5 || centroid.x > resolution[1] - 5) centroid.x = frisbeeX[i - 2];
				if (centroid.y < 5 || centroid.y > resolution[0] - 5) centroid.y = frisbeeY[i - 2];

				restrictCentroid(&centroid, &prevWidthBy2, &prevHeightBy2, &widthBy2, &heightBy2, &windowWidthChange, &windowHeightChange, resolution);

				Rect region_of_interest;

				if (goingUp)
					region_of_interest = Rect((centroid.x - widthBy2), (centroid.y - heightBy2), 2 * widthBy2, 2 * heightBy2);
				else
					region_of_interest = Rect((centroid.x - widthBy2), centroid.y-frisbeeHeight, 2 * widthBy2, 2 * heightBy2);

				try{
					image[i + k - 1] = image_[i + k - 1](region_of_interest);
				}catch(exception& e){
					cout<<"Exception For 1:\t"<<e.what()<<"\t"<<k<<endl;
					outputFile <<"ROI in while 1" <<region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
					cout<<(centroid.x - widthBy2)<<"\t"<<(centroid.y - heightBy2)<<"\t"<<2 * widthBy2<<"\t"<<2 * heightBy2<<endl;
					cout<<region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
					if (!restart)
						waitForStart = true;
					break;
				}

				Mat frame_red_bw(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
				Mat frame_red_diff(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
				Mat frame_greenPlane(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);

				if (useRGB)
					getRed(image[i + k - 1], frame_red_bw, frame_red_diff, frame_greenPlane, minThreshColor, maxThreshColor, minThreshGreen, maxThreshGreen, kernel);
				else{
					getHSV(image[i + k - 1], frame_red_bw, minThreshHSV, maxThreshHSV, kernel);
					//int thresh[] = { 0, 90};
					// if (frisbeeArea > 100)
					// 	maxThreshHSV[1] = getHSVHist(image[i+k-1], frame_red_bw, thresh, frisbeeArea, kernel);
					// else
					// 	maxThreshHSV[1] = getHSVHist(image[i+k-1], frame_red_bw, thresh, 200, kernel);
					//if (frisbeeArea > 100)
					//	maxThreshHSV[1] = getHSVHistHV(image[i+k-1], frame_red_bw, frisbeeArea+300, kernel);
					//else
					//	maxThreshHSV[1] = getHSVHistHV(image[i+k-1], frame_red_bw, 400, kernel);
				}

				Blob* windowBlobs = new Blob;
				getBlobProps(frame_red_bw, windowBlobs);
				int numberBlobs = (windowBlobs->centroids).size();

				Point predictedCentroid = Point(0, 0);

				if (numberBlobs < 1) {
					if(i == 1 || i == 2){
						
						int codec = CV_FOURCC('M', 'J', 'P', 'G');
						double fps = 30;
						VideoWriter writer;
						stringstream convert;
						convert << v;
						string temp = convert.str();
						string videoName = "video3" + temp + ".avi";
						writer.open(videoName, codec, fps, image_[0].size(), true);
						try{
							for (int m=3; m<15; m++){
								transpose(image_[m], image_[m]);
								flip(image_[m], image_[m], 1);
							}

							for (int m = 3; m < i+k-1; m++) {
								if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
									writer.write(image_[m]);
								else
									continue;
								//cout << "Written frame " << i << endl;
								image_[m].release();
							}
							for (int m = i+k-1; m < nCount; m++) {
								transpose(image_[m], image_[m]);
								flip(image_[m], image_[m], 1);
								if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
									writer.write(image_[m]);
								else
									continue;
								//cout << "Written frame " << i << endl;
								image_[m].release();
							}

						}catch (exception& e){
							cout<<"Video Write Error "<<i<<endl;
							cout<<e.what()<<endl;
							outputFile<<e.what()<<endl;
							v++;
							while(!captured && !restart);
							if (!restart){
								cout<<"Detection Fail in 1,1"<<endl;
								waitForStart = true;
								break;
							}
							else
								break;
						}
							
						while(!captured && !restart);
						if (!restart){
							cout<<"Detection Fail in 1,1"<<endl;
							outputFile<<"Detection Fail in 1,1"<<endl;
							waitForStart = true;
							v++;
							break;
						}
						else{
							v++;
							break;
						}
					}
					predictedCentroid.x = centroid.x + CalcMedian(changeX);
					predictedCentroid.y = centroid.y + CalcMedian(changeY);
					centroid.x = predictedCentroid.x;
					centroid.y = predictedCentroid.y;
					changeX.push_back(changeX[i - 1]);
					changeY.push_back(changeY[i - 1]);
					frisbeeX.push_back(frisbeeX[i - 1]);
					frisbeeY.push_back(frisbeeY[i - 1]);
					cout << "Centroid Prediction No" << endl;
					minThreshColor -= 5;
					defaultMinThresh -= 5;
					maxThreshColor -= 5;
					defaultMaxThresh -= 5;
					maxThreshGreen += 10;
					defaultMaxThreshGreen += 10;
				}

				else {
					if (i == 1 || i == 2) {
						minimum = 0;
						for (int j = 1; j < numberBlobs; j++) {
							if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
								&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
								((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200))&& abs((windowBlobs->dimension)[j][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[j][0] - frisbeeWidth)<20)
								minimum = j;
						}
						if (abs((windowBlobs->dimension)[minimum][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[minimum][0] - frisbeeWidth)<20){
							frisbeeX.push_back(centroid.x + (windowBlobs->centroids)[minimum].x - widthBy2);
							frisbeeY.push_back(centroid.y + (windowBlobs->centroids)[minimum].y - heightBy2);
							changeX.push_back(frisbeeX[i] - centroid.x);
							changeY.push_back(frisbeeY[i] - centroid.y);
							centroid.x = frisbeeX[i];
							centroid.y = frisbeeY[i];
							frisbeeWidth = (windowBlobs->dimension)[minimum][0];
							frisbeeHeight = (windowBlobs->dimension)[minimum][1];
							frisbeeArea = (windowBlobs->contourareas)[minimum];						
							circle(frame_red_bw, (windowBlobs->centroids)[minimum], 5, 255, -1, 8, 0);
							imwrite("Frisbee1.jpg", frame_red_bw);
						}
						else{
							int codec = CV_FOURCC('M', 'J', 'P', 'G');
							double fps = 30;
							VideoWriter writer;
							stringstream convert;
							convert << v;
							string temp = convert.str();
							string videoName = "video3" + temp + ".avi";
							writer.open(videoName, codec, fps, image_[0].size(), true);
							try{
								cout<<"Video is downloaded in 1,2"<<endl;
								for (int m=3; m<15; m++){
									transpose(image_[m], image_[m]);
									flip(image_[m], image_[m], 1);
								}

								for (int m = 3; m < i+k-1; m++) {
									if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
										writer.write(image_[m]);
									else
										continue;
									//cout << "Written frame " << i << endl;
									image_[m].release();
								}
								for (int m = i+k-1; m < nCount; m++) {
									transpose(image_[m], image_[m]);
									flip(image_[m], image_[m], 1);
									if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
										writer.write(image_[m]);
									else
										continue;
									//cout << "Written frame " << i << endl;
									image_[m].release();
								}
							}catch (exception& e){
								cout<<"Video Write Error "<<i<<endl;
								cout<<e.what()<<endl;
								outputFile<<e.what()<<endl;
								v++;
								while(!captured && !restart);
								if (!restart){
									cout<<"Detection Fail in 1,2"<<endl;
									waitForStart = true;
									break;
								}
								else
									break;

							}
							v++;
							while(!captured && !restart);
							if (!restart){
								cout<<"Detection Fail in 1,2"<<endl;
								waitForStart = true;
								break;
							}
							else
								break;
						}
						//cout << "Centroid from i=1" << endl;
						// cout << "Contour Area\t" << (windowBlobs->contourareas)[minimum] << endl;
						// cout << "Contour Eccentricity\t" << (windowBlobs->eccentricity)[minimum] << endl;
					}
					else {
						vector<Point> estimatedCentroid;
						vector<int> errorX;
						vector<int> errorY;
						predictedCentroid.x = centroid.x + CalcMedian(changeX);
						predictedCentroid.y = centroid.y + CalcMedian(changeY);

						for (int j = 0; j < numberBlobs; j++) {
							if (!goingUp)
								estimatedCentroid.push_back(Point(centroid.x + (windowBlobs->centroids)[j].x - widthBy2,
								centroid.y + (windowBlobs->centroids)[j].y-frisbeeHeight));
							else
								estimatedCentroid.push_back(Point(centroid.x + (windowBlobs->centroids)[j].x - widthBy2,
								centroid.y + (windowBlobs->centroids)[j].y - heightBy2));
							errorX.push_back(estimatedCentroid[j].x - predictedCentroid.x);
							errorY.push_back(estimatedCentroid[j].y - predictedCentroid.y);
						}

						minimum = 0;
						for (int j = 1; j < numberBlobs; j++) {
							if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
								&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
								((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)))
								minimum = j;
						}

						//Mat frame_cont = Mat::zeros(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
						//drawContours(frame_cont, (windowBlobs->contours), minimum, 255, -1);

						int m = 0;
						if (frisbeeArea > 4000)
							m = 4000;
						else if (frisbeeArea > 2000)
							m = 2000;
						else if (frisbeeArea > 1000)
							m = 1000;
						else if (frisbeeArea > 500)
							m = 500;
						else
							m = 200;

						decreaseArea = false;

						int min2 = 0;
						if (minimum == 0)
							min2 = 1;
						for (int j = 1; j < windowBlobs->contourareas.size(); j++){
							if (j == minimum)
								continue;
							if (abs(windowBlobs->centroids[minimum].x - windowBlobs->centroids[j].x) < frisbeeWidth &&
								abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[j].y) < abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[min2].y))
								min2 = j;
						}

						if (i > 5 && (windowBlobs->centroids).size() > 1 && abs(windowBlobs->centroids[minimum].x - windowBlobs->centroids[min2].x) < frisbeeWidth &&
							abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[min2].y) < frisbeeHeight){
							(windowBlobs->contourareas)[minimum] += windowBlobs->contourareas[min2];
							decreaseArea = true;
							//cout << "Final Centroid" << (windowBlobs->centroids)[minimum].x << "\t" << (windowBlobs->centroids)[minimum].y << endl;
						}

						bool useMethod1 = true;
						if (!goingUp)
							useMethod1 = abs(errorX[minimum]) < 25 & abs(errorY[minimum]) < 70;
						else if (i > 3)
							useMethod1 = abs(errorX[minimum]) < 25 & abs(errorY[minimum]) < 90;

						if (abs(frisbeeArea - (windowBlobs->contourareas)[minimum]) < m  && methodUsed == 1 && useMethod1) {

							centroid.x = estimatedCentroid[minimum].x;
							if (!decreaseArea)
								centroid.y = estimatedCentroid[minimum].y;
							else{
								if (goingUp)
									centroid.y = (windowBlobs->centroids[minimum].y + windowBlobs->centroids[min2].y) / 2 - heightBy2 + centroid.y;
								else
									centroid.y = (windowBlobs->centroids[minimum].y + windowBlobs->centroids[min2].y) / 2 + centroid.y - frisbeeHeight;
							}

							frisbeeX.push_back(centroid.x);
							frisbeeY.push_back(centroid.y);
							changeX.push_back(frisbeeX[i] - frisbeeX[i - 1]);
							changeY.push_back(frisbeeY[i] - frisbeeY[i - 1]);
							frisbeeArea = (windowBlobs->contourareas)[minimum];
							/*
							if (useRGB){
								minThreshColor = 255; maxThreshColor = 0;
								minThreshGreen = 255; maxThreshGreen = 0;
								for (int j = 0; j < image[i + k - 1].rows; j++) {
									unsigned char* E = frame_cont.ptr<uchar>(j);
									unsigned char* F = frame_red_diff.ptr<uchar>(j);
									unsigned char* G = frame_greenPlane.ptr<uchar>(j);
									for (int u = 0; u < image[i + k - 1].cols; u++){
										if (E[u] == 255){
											int pixval2 = G[u];
											int pixval = F[u];
											if (pixval > maxThreshColor) maxThreshColor = pixval;
											if (pixval < minThreshColor) minThreshColor = pixval;
											if (pixval2 > maxThreshGreen) maxThreshGreen = pixval2;
											if (pixval2 < minThreshGreen) minThreshGreen = pixval2;
										}
									}
								}


								constrain(defaultMinThresh, 255, &minThreshColor);
								constrain(0, defaultMaxThresh, &maxThreshColor);
								constrain(defaultMinThreshGreen, 255, &minThreshGreen);
								constrain(0, defaultMaxThreshGreen, &maxThreshGreen);
								if ((maxThreshColor - minThreshColor) <= 5){
									minThreshColor -= 2;
									maxThreshColor += 2;
								}

								if ((maxThreshGreen - minThreshGreen) <= 5){
									minThreshGreen -= 2;
									maxThreshGreen += 2;
								}
							}
							*/
							//cout<<"Normal Centroid"<<endl;
							frisbeeWidth = (windowBlobs->dimension)[minimum][0];
							frisbeeHeight = (windowBlobs->dimension)[minimum][1];

						}
						else if (methodUsed == 1)
							methodUsed = 2;

						if (numberBlobs >= 2 && methodUsed == 3) {
							int idx = 0;
							for (int j = 1; j < numberBlobs; j++) {
								if ((0.9*abs(errorX[j]) + 0.1*abs(errorY[j])) < (0.8*abs(errorX[idx]) + 0.1*abs(errorY[idx])) &&
									(windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
									((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)))
									idx = j;
							}

							if ((abs(0.9*errorX[idx]) + 0.1*abs(errorY[idx])) < 25 && (windowBlobs->eccentricity)[idx] > eccentricityThresh &&
								abs(windowBlobs->contourareas[idx] - frisbeeArea) < m && (((windowBlobs->angle)[idx]>340 || (windowBlobs->angle)[idx]<200) ||
						((windowBlobs->angle)[idx]>160 && (windowBlobs->angle)[idx]<200))) {
								centroid.x = estimatedCentroid[idx].x;
								centroid.y = estimatedCentroid[idx].y;
								frisbeeX.push_back(centroid.x);
								frisbeeY.push_back(centroid.y);
								changeX.push_back(frisbeeX[i] - frisbeeX[i - 1]);
								changeY.push_back(frisbeeY[i] - frisbeeY[i - 1]);
								frisbeeArea = (windowBlobs->contourareas)[idx];
								methodUsed = 1;
								//cout << "Centroid after coming out of red" << endl;
							}
							else {
								methodUsed = 2;
							}
						}

						else if (methodUsed == 3) {
							methodUsed = 2;
						}

						if (methodUsed == 2) {
							centroid.x = predictedCentroid.x;
							centroid.y = predictedCentroid.y;
							changeX.push_back(changeX[changeX.size() - 1]);
							changeY.push_back(changeY[changeY.size() - 1]);
							frisbeeX.push_back(frisbeeX[i - 1]);
							frisbeeY.push_back(frisbeeY[i - 1]);


							if (decreaseArea){
								decreaseArea = false;
								methodUsed = 1;
							}
							methodUsed = 3;
							//cout << "Centroid Prediction" << endl;
							heightBy2 += 10;
						}
					}
				}

				if (widthBy2 > 100) widthBy2 -= 10;
				if (heightBy2 > 100) heightBy2 -= 10;
				if (windowWidthChange) {
					widthBy2 = prevWidthBy2;
					windowWidthChange = false;
				}
				if (windowHeightChange) {
					heightBy2 = prevHeightBy2;
					windowHeightChange = false;
				}

				circle(image_[i + k - 1], centroid, 5, 0, -1, 8, 0);
				if (i==2){
					circle(frame_red_bw, (windowBlobs->centroids)[minimum], 5, 255, -1, 8, 0);
					imwrite("Frisbee2.jpg", frame_red_bw);
				}
				image[i + k - 1].release();
			}
		}
	}

	omp_set_num_threads(4);
	
	if (restart){
		resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
						&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
		goto START;
	}

	if (waitForStart){
		while(1){
			if (serialDataAvail(fd)) {
					z = serialGetchar(fd);
					if (z == 5 || z == 6) {
						poleUsed = z-4;
						serialFlush(fd);
						resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
							&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
						goto START;
				}
			}
		}
	}

	while (!frisbeeFound){

		if (serialDataAvail(fd)) {
			z = serialGetchar(fd);
			if (z == 5 || z == 6) {
				poleUsed = z-4;
				serialFlush(fd);
				resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
					&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
				goto START;
			}
		}
		
		if (k >= 32){

			outputFile << "Not Detected"<<endl;
			int codec = CV_FOURCC('M', 'J', 'P', 'G');
			double fps = 30;
			VideoWriter writer;
			stringstream convert;
			convert << v;
			string temp = convert.str();
			string videoName = "video3" + temp + ".avi";
			writer.open(videoName, codec, fps, image_[0].size(), true);
			for (int m=3; m<15; m++){
				transpose(image_[m], image_[m]);
				flip(image_[m], image_[m], 1);
			}

			for (int m = 3; m < 32; m++) {
				if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
					writer.write(image_[m]);
				else
					continue;
				//cout << "Written frame " << i << endl;
				image_[m].release();
			}
			v++;

			while(1){
				if (serialDataAvail(fd)) {
					z = serialGetchar(fd);
					if (z == 5 || z == 6) {
						poleUsed = z-4;
						serialFlush(fd);
						resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
							&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
						goto START;
					}
				}
			}
		}

		while (image_[k].empty());
		transpose(image_[k], image_[k]);
		flip(image_[k], image_[k], 1);

		restrictCentroid(&centroid, &prevWidthBy2, &prevHeightBy2, &widthBy2, &heightBy2, &windowWidthChange, &windowHeightChange, resolution);
		Rect region_of_interest = Rect((centroid.x - widthBy2), (centroid.y - heightBy2), 2 * widthBy2, 2 * heightBy2);
		
		try{
			image[k] = image_[k](region_of_interest);
		}catch(exception& e){
			cout<<"Exception While 2:\t"<<e.what()<<"\t"<<k<<endl;
			outputFile<<"Exception While 2:\t"<<e.what()<<"\t"<<k<<endl;
			
			cout<<(centroid.x - widthBy2)<<"\t"<<(centroid.y - heightBy2)<<"\t"<<2 * widthBy2<<"\t"<<2 * heightBy2<<endl;
			cout << region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
			outputFile <<"ROI: " <<region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
			
			cout<<image_[k].rows<<" "<<image_[k].cols<<" "<<image_[k].channels()<<endl;
			k++;
			continue;
		}
		Mat frame_red_bw(image[k].rows, image[k].cols, CV_8UC1);
		Mat frame_red_diff(image[k].rows, image[k].cols, CV_8UC1);
		Mat frame_greenPlane(image[k].rows, image[k].cols, CV_8UC1);

		if (useRGB)
			getRed(image[k], frame_red_bw, frame_red_diff, frame_greenPlane, minThreshColor, maxThreshColor, minThreshGreen, maxThreshGreen, kernel);
		else{
			getHSV(image[k], frame_red_bw, minThreshHSV, maxThreshHSV, kernel);
			//int thresh[] = {0, 90};
			//int maxV = getHSVHist(image[k], frame_red_bw, thresh, 4000, kernel);
			//cout << "MaxV:\t"<<maxV<<endl;
			//int maxV = getHSVHistHV(image[k], frame_red_bw, 4000, kernel);
			// cout << "<----------------------------MaxV----------------------------->"<<endl;
			// cout << maxV << endl;
			// outputFile<<"MaxV:\t"<<maxV<<endl;

		}

		Blob* windowBlobs = new Blob;
		getBlobProps(frame_red_bw, windowBlobs);
		int numberBlobs = (windowBlobs->centroids).size();

		if (numberBlobs >= 1){
			minimum = 0;
			for (int j = 1; j < numberBlobs; j++) {
				if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
					&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
				((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)) && abs((windowBlobs->dimension)[j][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[j][0] - frisbeeWidth)<20)
					minimum = j;
			}
			//cout<<"Blobs Detected!\nArea: ";
			//for (int j=0; j<numberBlobs; j++){
			//	cout<<(windowBlobs->contourareas)[j]<<" ";
			//}
			//cout<<endl;
			if (abs(frisbeeArea - (windowBlobs->contourareas)[minimum]) < 1000 && (windowBlobs->eccentricity)[minimum] > eccentricityThresh && 
				(((windowBlobs->angle)[minimum]>340 || (windowBlobs->angle)[minimum]<20) || ((windowBlobs->angle)[minimum]>160 && (windowBlobs->angle)[minimum]<200))
				&& abs((windowBlobs->dimension)[minimum][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[minimum][0] - frisbeeWidth)<20){
				frisbeeX.push_back(centroid.x + (windowBlobs->centroids)[minimum].x - widthBy2);
				frisbeeY.push_back(centroid.y + (windowBlobs->centroids)[minimum].y - heightBy2);
				centroid.x = frisbeeX[0];
				centroid.y = frisbeeY[0];
				frisbeeArea = (windowBlobs->contourareas)[minimum];
				frisbeeFound = true;
				cout << "Found on iteration " << k << " Area = " << frisbeeArea << endl;
				circle(image_[k], centroid, 5, 0, -1, 8, 0);
				imwrite("frisbee0.jpg", image_[k]);
			}
		}
		image[k].release();
		k++; 
	}

	if (id == 0) {
		widthBy2 = 200; 
		prevWidthBy2 = 200;
		heightBy2 = 200;
		prevHeightBy2 = 200;
		id = 1;
	}


	for (int i = id; i < nCount - k + 1; i++) {
		
		if (serialDataAvail(fd)) {
			z = serialGetchar(fd);
			if ((z == 5 || z == 6) && i<3) {
				poleUsed = z - 4;
				serialFlush(fd);
				resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
					&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
				goto START;
			}
		}

		if (i > 2 && centroid.y<300 && counter < 3){
			if (counter == 0)
				cout << "Threshold height reaches" << endl;
			counter += 1;
			frisbeeX.push_back(centroid.x + changeX[changeX.size() - 1]);
			frisbeeY.push_back(centroid.y);
			changeX.push_back(frisbeeX[frisbeeX.size()-1] - centroid.x);
			changeY.push_back(frisbeeY[frisbeeX.size()-1] - centroid.y);
			centroid.x = frisbeeX[frisbeeX.size() - 1];
			centroid.y = frisbeeY[frisbeeY.size() - 1];
			frisbeeArea = 500;
			frisbeeWidth -= 3 * 12;
			frisbeeHeight -= 3 * 1.5;
			continue;
		}
		
		while (image_[i + k - 1].empty());
		transpose(image_[i + k - 1], image_[i + k - 1]);
		flip(image_[i + k - 1], image_[i + k - 1], 1);

		if (centroid.x < 5 || centroid.x > resolution[1] - 5) centroid.x = frisbeeX[i - 2];
		if (centroid.y < 5 || centroid.y > resolution[0] - 5) centroid.y = frisbeeY[i - 2];

		restrictCentroid(&centroid, &prevWidthBy2, &prevHeightBy2, &widthBy2, &heightBy2, &windowWidthChange, &windowHeightChange, resolution);

		Rect region_of_interest;
		if (goingUp)
			region_of_interest = Rect((centroid.x - widthBy2), (centroid.y - heightBy2), 2 * widthBy2, 2 * heightBy2);
		else
			region_of_interest = Rect((centroid.x - widthBy2), centroid.y - frisbeeHeight, 2 * widthBy2, 2 * heightBy2);

		try{
			image[i + k - 1] = image_[i + k - 1](region_of_interest);
		}catch(exception& e){
			cout<<"Exception For 2:\t"<<e.what()<<"\t"<<k<<endl;
			outputFile<<"Exception For 2:\t"<<e.what()<<"\t"<<k<<endl;
			
			cout<<(centroid.x - widthBy2)<<"\t"<<(centroid.y - heightBy2)<<"\t"<<2 * widthBy2<<"\t"<<2 * heightBy2<<endl;
			outputFile <<"ROI: " <<region_of_interest.x << " " << region_of_interest.y << " " << region_of_interest.width << " " << region_of_interest.height<<endl;
			
			continue;
		}
		Mat frame_red_bw(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
		Mat frame_red_diff(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
		Mat frame_greenPlane(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);

		if (useRGB)
			getRed(image[i + k - 1], frame_red_bw, frame_red_diff, frame_greenPlane, minThreshColor, maxThreshColor, minThreshGreen, maxThreshGreen, kernel);
		else{
			getHSV(image[i + k - 1], frame_red_bw, minThreshHSV, maxThreshHSV, kernel);
			//int thresh[] = { 0, 90 };
			// if (frisbeeArea > 100)
			// 	maxThreshHSV[1] = getHSVHist(image[i+k-1], frame_red_bw, thresh, frisbeeArea, kernel);
			// else 
			// 	maxThreshHSV[1] = getHSVHist(image[i+k-1], frame_red_bw, thresh, 200, kernel);
			//if (frisbeeArea > 100)
			//	maxThreshHSV[1] = getHSVHistHV(image[i+k-1], frame_red_bw, frisbeeArea+300, kernel);
			//else
			//	maxThreshHSV[1] = getHSVHistHV(image[i+k-1], frame_red_bw, 400, kernel);
		}

		Blob* windowBlobs = new Blob;
		getBlobProps(frame_red_bw, windowBlobs);
		int numberBlobs = (windowBlobs->centroids).size();

		Point predictedCentroid = Point(0, 0);

		if (numberBlobs < 1) {
			if (i == 1 || i == 2){
				
				int codec = CV_FOURCC('M', 'J', 'P', 'G');
				double fps = 30;
				VideoWriter writer;
				stringstream convert;
				convert << v;
				string temp = convert.str();
				string videoName = "video3" + temp + ".avi";
				writer.open(videoName, codec, fps, image_[0].size(), true);
				try{
					for (int m=3; m<15; m++){
						transpose(image_[m], image_[m]);
						flip(image_[m], image_[m], 1);
					}

					for (int m = 3; m < i+k-1; m++) {
						if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
							writer.write(image_[m]);
						else
							continue;
						//cout << "Written frame " << i << endl;
						image_[m].release();
					}
					for (int m = i+k-1; m < nCount; m++) {
						transpose(image_[m], image_[m]);
						flip(image_[m], image_[m], 1);
						if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
							writer.write(image_[m]);
						else
							continue;
						//cout << "Written frame " << i << endl;
						image_[m].release();
					}
				}catch (exception& e){
					cout<<"Video Write Error "<<i<<endl;
					cout<<e.what()<<endl;
					v++;
					while (1) {
						if (serialDataAvail(fd)) {
							cout<<"Detection Fail in 2,1"<<endl;
							outputFile<<"Detection Fail in 2,1"<<endl;

							z = serialGetchar(fd);
							if (z == 5 || z == 6) {
								poleUsed = z - 4;
								serialFlush(fd);
								resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
									&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
								goto START;
							}
							serialFlush(fd);
						}
					}
				}
				v++;
				
				while (1) {
					if (serialDataAvail(fd)) {
						outputFile<<"Detection Fail in 2,1"<<endl;
						cout<<"Detection Fail in 2,1"<<endl;
						z = serialGetchar(fd);
						if (z == 5 || z == 6) {
							poleUsed = z - 4;
							serialFlush(fd);
							resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
								&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
							goto START;
						}
						serialFlush(fd);
					}
				}
			}
			predictedCentroid.x = centroid.x + CalcMedian(changeX);
			predictedCentroid.y = centroid.y + CalcMedian(changeY);
			centroid.x = predictedCentroid.x;
			centroid.y = predictedCentroid.y;
			changeX.push_back(changeX[i - 1]);
			changeY.push_back(changeY[i - 1]);
			frisbeeX.push_back(frisbeeX[i - 1]);
			frisbeeY.push_back(frisbeeY[i - 1]);
			minThreshColor -= 5;
			defaultMinThresh -= 5;
			maxThreshColor -= 5;
			defaultMaxThresh -= 5;
			maxThreshGreen += 10;
			defaultMaxThreshGreen += 10;
		}

		else {
			if (i == 1 || i == 2) {
				minimum = 0;
				for (int j = 1; j < numberBlobs; j++) {
					if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
						&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
						((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200))&& abs((windowBlobs->dimension)[j][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[j][0] - frisbeeWidth)<20)
						minimum = j;
				}
				if (abs((windowBlobs->dimension)[minimum][1] - frisbeeHeight)<50 && abs((windowBlobs->dimension)[minimum][0] - frisbeeWidth)<20){
					frisbeeX.push_back(centroid.x + (windowBlobs->centroids)[minimum].x - widthBy2);
					frisbeeY.push_back(centroid.y + (windowBlobs->centroids)[minimum].y - heightBy2);
					changeX.push_back(frisbeeX[i] - centroid.x);
					changeY.push_back(frisbeeY[i] - centroid.y);
					centroid.x = frisbeeX[i];
					centroid.y = frisbeeY[i];
					frisbeeWidth = (windowBlobs->dimension)[minimum][0];
					frisbeeHeight = (windowBlobs->dimension)[minimum][1];
					frisbeeArea = (windowBlobs->contourareas)[minimum];						
					circle(frame_red_bw, (windowBlobs->centroids)[minimum], 5, 255, -1, 8, 0);
					imwrite("Frisbee1.jpg", frame_red_bw);
				}
				
				else{
					
					int codec = CV_FOURCC('M', 'J', 'P', 'G');
					double fps = 30;
					VideoWriter writer;
					stringstream convert;
					convert << v;
					string temp = convert.str();
					string videoName = "video3" + temp + ".avi";
					writer.open(videoName, codec, fps, image_[0].size(), true);
					try{
						for (int m=3; m<15; m++){
							transpose(image_[m], image_[m]);
							flip(image_[m], image_[m], 1);
						}

						for (int m = 3; m < i+k-1; m++) {
							if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
								writer.write(image_[m]);
							else
								continue;
							//cout << "Written frame " << i << endl;
							image_[m].release();
						}
						for (int m = i+k-1; m < nCount; m++) {
							transpose(image_[m], image_[m]);
							flip(image_[m], image_[m], 1);
							if (image_[m].channels() == 3 && image_[m].rows == resolution[0] && image_[m].cols == resolution[1])
								writer.write(image_[m]);
							else
								continue;
							//cout << "Written frame " << i << endl;
							image_[m].release();
						}
					}catch (exception& e){
						cout<<"Video Write Error "<<i<<endl;
						cout<<e.what()<<endl;
						outputFile<<e.what()<<endl;
						v++;
						while (1) {
							if (serialDataAvail(fd)) {
								cout<<"Detection Fail in 2,2"<<endl;
								outputFile<<"Detection Fail in 2,2"<<endl;
								z = serialGetchar(fd);
								if (z == 5 || z == 6) {
									poleUsed = z - 4;
									serialFlush(fd);
									resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
										&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
									goto START;
								}
								serialFlush(fd);
							}
						}
					}
					v++;
					
					while (1) {
						if (serialDataAvail(fd)) {
							cout<<"Detection Fail in 2,2"<<endl;
							outputFile<<"Detection Fail in 2,2"<<endl;
							
							z = serialGetchar(fd);
							if (z == 5 || z == 6) {
								poleUsed = z - 4;
								serialFlush(fd);
								resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
									&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
								goto START;
							}
							serialFlush(fd);
						}
					}
				}
			}
			else {

				vector<Point> estimatedCentroid;
				vector<int> errorX;
				vector<int> errorY;
				predictedCentroid.x = centroid.x + CalcMedian(changeX);
				predictedCentroid.y = centroid.y + CalcMedian(changeY);

				for (int j = 0; j < numberBlobs; j++) {
					if (!goingUp)
						estimatedCentroid.push_back(Point(centroid.x + (windowBlobs->centroids)[j].x - widthBy2,
						centroid.y + (windowBlobs->centroids)[j].y));
					else
						estimatedCentroid.push_back(Point(centroid.x + (windowBlobs->centroids)[j].x - widthBy2,
						centroid.y + (windowBlobs->centroids)[j].y - heightBy2));
					errorX.push_back(estimatedCentroid[j].x - predictedCentroid.x);
					errorY.push_back(estimatedCentroid[j].y - predictedCentroid.y);
				}


				minimum = 0;
				for (int j = 1; j < numberBlobs; j++) {
					if (abs(frisbeeArea - (windowBlobs->contourareas)[j]) < abs(frisbeeArea - (windowBlobs->contourareas)[minimum])
						&& (windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
						((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)))
						minimum = j;
				}

				//Mat frame_cont = Mat::zeros(image[i + k - 1].rows, image[i + k - 1].cols, CV_8UC1);
				//drawContours(frame_cont, (windowBlobs->contours), minimum, 255, -1);

				int m = 0;
				if (frisbeeArea > 4000)
					m = 4000;
				else if (frisbeeArea > 2000)
					m = 2000;
				else if (frisbeeArea > 1000)
					m = 1000;
				else if (frisbeeArea > 500)
					m = 500;
				else
					m = 200;

				decreaseArea = false;

				int min2 = 0;
				if (minimum == 0)
					min2 = 1;
				for (int j = 1; j < windowBlobs->contourareas.size(); j++){
					if (j == minimum)
						continue;
					if (abs(windowBlobs->centroids[minimum].x - windowBlobs->centroids[j].x) < frisbeeWidth &&
						abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[j].y) < abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[min2].y))
						min2 = j;
				}

				if (i > 5 && (windowBlobs->centroids).size() > 1 && abs(windowBlobs->centroids[minimum].x - windowBlobs->centroids[min2].x) < frisbeeWidth &&
					abs(windowBlobs->centroids[minimum].y - windowBlobs->centroids[min2].y) < frisbeeHeight){
					(windowBlobs->contourareas)[minimum] += windowBlobs->contourareas[min2];
					decreaseArea = true;
					//cout << "Final Centroid" << (windowBlobs->centroids)[minimum].x << "\t" << (windowBlobs->centroids)[minimum].y << endl;
				}

				bool useMethod1 = true;
				if (!goingUp)
					useMethod1 = abs(errorX[minimum]) < 25 & abs(errorY[minimum]) < 70;
				else if (i > 3)
					useMethod1 = abs(errorX[minimum]) < 25 & abs(errorY[minimum]) < 90;

				if (abs(frisbeeArea - (windowBlobs->contourareas)[minimum]) < m  && methodUsed == 1 && useMethod1) {

					centroid.x = estimatedCentroid[minimum].x;
					if (!decreaseArea)
						centroid.y = estimatedCentroid[minimum].y;
					else{
						if (goingUp)
							centroid.y = (windowBlobs->centroids[minimum].y + windowBlobs->centroids[min2].y) / 2 - heightBy2 + centroid.y;
						else
							centroid.y = (windowBlobs->centroids[minimum].y + windowBlobs->centroids[min2].y) / 2 + centroid.y - frisbeeHeight;
					}

					frisbeeX.push_back(centroid.x);
					frisbeeY.push_back(centroid.y);
					changeX.push_back(frisbeeX[i] - frisbeeX[i - 1]);
					changeY.push_back(frisbeeY[i] - frisbeeY[i - 1]);
					frisbeeArea = (windowBlobs->contourareas)[minimum];

					/*
					if (useRGB){
						minThreshColor = 255; maxThreshColor = 0;
						minThreshGreen = 255; maxThreshGreen = 0;
						for (int j = 0; j < image[i + k - 1].rows; j++) {
							unsigned char* E = frame_cont.ptr<uchar>(j);
							unsigned char* F = frame_red_diff.ptr<uchar>(j);
							unsigned char* G = frame_greenPlane.ptr<uchar>(j);
							for (int u = 0; u < image[i + k - 1].cols; u++){
								if (E[u] == 255){
									int pixval2 = G[u];
									int pixval = F[u];
									if (pixval > maxThreshColor) maxThreshColor = pixval;
									if (pixval < minThreshColor) minThreshColor = pixval;
									if (pixval2 > maxThreshGreen) maxThreshGreen = pixval2;
									if (pixval2 < minThreshGreen) minThreshGreen = pixval2;
								}
							}
						}

						constrain(defaultMinThresh, 255, &minThreshColor);
						constrain(0, defaultMaxThresh, &maxThreshColor);
						constrain(defaultMinThreshGreen, 255, &minThreshGreen);
						constrain(0, defaultMaxThreshGreen, &maxThreshGreen);
						if ((maxThreshColor - minThreshColor) <= 5){
							minThreshColor -= 2;
							maxThreshColor += 2;
						}

						if ((maxThreshGreen - minThreshGreen) <= 5){
							minThreshGreen -= 2;
							maxThreshGreen += 2;
						}
					}
					*/
					//cout<<"Normal Centroid"<<endl;
					frisbeeWidth = (windowBlobs->dimension)[minimum][0];
					frisbeeHeight = (windowBlobs->dimension)[minimum][1];
				}
				else if (methodUsed == 1)
					methodUsed = 2;

				if (numberBlobs >= 2 && methodUsed == 3) {
					int idx = 0;
					for (int j = 1; j < numberBlobs; j++) {
						if ((0.9*abs(errorX[j]) + 0.1*abs(errorY[j])) < (0.8*abs(errorX[idx]) + 0.1*abs(errorY[idx])) &&
							(windowBlobs->eccentricity)[j] > eccentricityThresh && (((windowBlobs->angle)[j]>340 || (windowBlobs->angle)[j]<20) ||
							((windowBlobs->angle)[j]>160 && (windowBlobs->angle)[j]<200)))
							idx = j;
					}

					if ((abs(0.9*errorX[idx]) + 0.1*abs(errorY[idx])) < 25 && (windowBlobs->eccentricity)[idx] > eccentricityThresh &&
						abs(windowBlobs->contourareas[idx] - frisbeeArea) < m && (((windowBlobs->angle)[idx]>340 || (windowBlobs->angle)[idx]<200) ||
						((windowBlobs->angle)[idx]>160 && (windowBlobs->angle)[idx]<200))) {
						centroid.x = estimatedCentroid[idx].x;
						centroid.y = estimatedCentroid[idx].y;
						frisbeeX.push_back(centroid.x);
						frisbeeY.push_back(centroid.y);
						changeX.push_back(frisbeeX[i] - frisbeeX[i - 1]);
						changeY.push_back(frisbeeY[i] - frisbeeY[i - 1]);
						frisbeeArea = (windowBlobs->contourareas)[idx];
						methodUsed = 1;
						
						//cout << "Centroid after coming out of red" << endl;
					}
					else {
						methodUsed = 2;
					}
				}

				else if (methodUsed == 3) {
					methodUsed = 2;
				}

				if (methodUsed == 2) {
					centroid.x = predictedCentroid.x;
					centroid.y = predictedCentroid.y;
					changeX.push_back(changeX[changeX.size() - 1]);
					changeY.push_back(changeY[changeY.size() - 1]);
					frisbeeX.push_back(frisbeeX[i - 1]);
					frisbeeY.push_back(frisbeeY[i - 1]);


					if (decreaseArea){
						decreaseArea = false;
						methodUsed = 1;
					}
					methodUsed = 3;
					//cout << "Centroid Prediction" << endl;
					heightBy2 += 10;
				}
			}
		}

		if (widthBy2 > 100) widthBy2 -= 10;
		if (heightBy2 > 100) heightBy2 -= 10;
		if (windowWidthChange) {
			widthBy2 = prevWidthBy2;
			windowWidthChange = false;
		}
		if (windowHeightChange) {
			heightBy2 = prevHeightBy2;
			windowHeightChange = false;
		}
		if (changeY[changeY.size() - 1] > -7 && changeY[changeY.size() - 1] < 7){
			int maxH = frisbeeY[frisbeeY.size() - 1];
			stringstream convert;
			convert << maxH;
			string tempstr = convert.str();
			char* maxHeight = &tempstr[0u];
			//cout << "Max Height" << maxH<<endl;
			//serialPuts(fd, maxHeight);
		}

		circle(image_[i + k - 1], centroid, 5, 0, -1, 8, 0);
		if (i==2){
			circle(frame_red_bw, (windowBlobs->centroids)[minimum], 5, 255, -1, 8, 0);
			imwrite("Frisbee2.jpg", frame_red_bw);
		}
		image[i + k - 1].release();
		if(poleUsed == 1){
			if ((spotCentroidI.y - centroid.y) < 20){
				circle(image_[i + k - 1], centroid, 15, 150, -1, 8, 0);
				lastFrame = i + k - 1;
				break;
			}
		}
		else{
			if ((spotCentroidII.y - centroid.y) < 20){
				circle(image_[i + k - 1], centroid, 15, 0, -1, 8, 0);
				lastFrame = i + k - 1;
				break;
			}
		}
		if (i > 16){
			lastFrame = i+k-1;
			break;
		}
	}

	transpose(image_[1], image_[1]);
	flip(image_[1], image_[1], 1);

	transpose(image_[lastFrame+1], image_[lastFrame+1]);
	flip(image_[lastFrame+1], image_[lastFrame+1], 1);
	
	try{
		Rect poleRegion;
		if (poleUsed == 1)
			poleRegion = Rect(50, 760, 400, 40);
		else
			poleRegion = Rect(50, 760, 400, 40);
		image[0] = image_[0];
		image[1] = image_[1];
		image[lastFrame+1] = image_[lastFrame+1](poleRegion);
	} catch(exception& e){
		cout<<"Exception Pole: "<<e.what()<<endl;
		outputFile<<"Exception Pole: "<<e.what()<<endl;
		
		goto START;
	}
	int hist[256];
	for (int l=0; l<256; l++)
		hist[l] = 0;
	cvtColor(image[lastFrame+1], image[lastFrame+1], CV_BGR2HSV_FULL, 0);
	Mat saturation(image[lastFrame+1].rows, image[lastFrame+1].cols, CV_8UC1);
	Mat value(image[lastFrame+1].rows, image[lastFrame+1].cols, CV_8UC1);
	Mat I_gray_bw(image[lastFrame+1].rows, image[lastFrame+1].cols, CV_8UC1);
	int from_to[] = { 1, 0 };
	mixChannels(&image[lastFrame+1], 1, &saturation, 1, from_to, 1);
	from_to[0] = 2;
	mixChannels(&image[lastFrame+1], 1, &value, 1, from_to, 1);
	Mat I_bw1(image[lastFrame+1].rows, image[lastFrame+1].cols, CV_8UC1);
	Mat I_bw2(image[lastFrame+1].rows, image[lastFrame+1].cols, CV_8UC1);
	threshold(value, I_bw1, 0, 255, THRESH_BINARY);
	threshold(value, I_bw2, 255, 255, THRESH_BINARY_INV);
	bitwise_and(I_bw1, I_bw2, value, noArray());
	for (int l = 0; l < saturation.rows; l++) {
		unsigned char* r = saturation.ptr<uchar>(l);
		for (int u = 0; u < saturation.cols; u++)
				hist[r[u]] += 1;
	}

	int maxS = 0;
	int sum = 0;

	while(sum<1500){
		sum += hist[maxS];
		maxS++;
	}
	cout<<maxS<<endl;
	if (poleUsed == 1)
		getHist(saturation, I_gray_bw, value, maxS, kernelWhiteI);
	else
		getHist(saturation, I_gray_bw, value, maxS, kernelWhiteII);

	Mat I_gray_bw_ = I_gray_bw.clone();
	Rectangle* spot = new Rectangle();
	getRectProps(I_gray_bw, spot);
	cout<<"Pole is being detected"<<endl;
	
	imwrite("image0.jpg", image[lastFrame+1]);
	if ((spot->contours).size() > 0){
		//cout<<"<---------Spot found----------->"<<endl;
		int spotPos = 0;
		//cout<<"Pole Coord"<<endl;
		int spotWidth = spotWidthI;
		int spotHeight = spotHeightII;
		if (poleUsed == 1){
			spotWidth = spotWidthI;
			spotHeight = spotHeightI;
		}
		else{
			spotWidth = spotWidthII;
			spotHeight = spotHeightII;
		}

		for (int i = 1; i < spot->contours.size(); i++){
			//cout<<(spot->centroids)[i].x<<"\t"<<(spot->centroids)[i].y<<endl;
			//cout<<(spot->dimension)[i][0]<<"\t"<<(spot->dimension)[i][1]<<(spot->dimension)[i][2]<<endl;
			if ((spot->centroids)[i].x > minXSpot && (spot->centroids)[i].x < maxXSpot &&
			 (0.3*abs((spot->dimension)[i][0]-spotWidth)+0.7*abs((spot->dimension)[i][1]-spotHeight)) < 
			 (0.3*abs((spot->dimension)[spotPos][0]-spotWidth)+0.7*abs((spot->dimension)[spotPos][1]-spotHeight)))
				spotPos = i;
		}
		circle(I_gray_bw, (spot->centroids)[spotPos], 10, 255, -1, 8, 0);
		imwrite("spot2.jpg", I_gray_bw);
		circle(image[lastFrame+1], (spot->centroids)[spotPos], 10, 150, -1, 8, 0);
		imwrite("image0.jpg", image[lastFrame+1]);
		int spotX = (spot->centroids)[spotPos].x+50;
		
		if (poleUsed == 1){
			spotCentroidI.x = spotX;
		}
		else{
			spotCentroidII.x = spotX;		
		}
		imwrite("spot3.jpg", I_gray_bw_);
	}


	int drift2 = 0;
	if(frisbeeFound){
		if (poleUsed == 1)
			drift2 = frisbeeX[frisbeeX.size() - 1] - spotCentroidI.x;
		else
			drift2 = frisbeeX[frisbeeX.size() - 1] - spotCentroidII.x;
	}

	unsigned char drift = 0;
	cout<<"Drift2:\t"<<drift2<<endl;
	int quotient = drift2+15;
	if (drift2<0)
		quotient = drift2-15;
	quotient = quotient/30;
	if (drift2 > -20 && drift2 < 30)
		drift = 10;
	else
		drift = 10+quotient;
	if (poleUsed == 1)
		circle(image_[30], spotCentroidI, 15, 0, -1, 8, 0);
	else
		circle(image_[30], spotCentroidII, 15, 0, -1, 8, 0);
	

	imwrite("Final.jpg", image_[30]);
	//if (drift<20 && drift>0)
	//	serialPutchar(fd, drift);
	cout << "Drift:\t" << (int)drift << "\t" << drift2 << endl;
	outputFile << "Drift:\t" << (int)drift << "\t" << drift2 << endl;

	clock_gettime(CLOCK_REALTIME, &gettime_now);
	time_difference = gettime_now.tv_nsec - start_time;
	if (time_difference < 0)
		time_difference += 1000000000;
	cout << "Time elapsed = " << time_difference << " nanoseconds." << endl;
	time(&timer_end); // get current time; same as: timer = time(NULL) 
	double secondsElapsed = difftime(timer_end, timer_begin);
	cout << secondsElapsed << " seconds for " << lastFrame << "  frames : FPS = " << (float)((float)(lastFrame) / secondsElapsed) << endl;

	outputFile << "Video Write "<<v<<endl<<endl<<endl;

	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	double fps = 30;
	VideoWriter writer;
	stringstream convert;
	convert << v;
	string temp = convert.str();
	string videoName = "video3" + temp + ".avi";
	writer.open(videoName, codec, fps, image_[0].size(), true);
	for (int i = 2; i < lastFrame; i++) {
		if (image_[i].channels() == 3 && image_[i].rows == resolution[0] && image_[i].cols == resolution[1])
			writer.write(image_[i]);
		else
			continue;
		//cout << "Written frame " << i << endl;
		image_[i].release();
	}
	v++;
	 
	while (1) {
		if (serialDataAvail(fd)) {
			z = serialGetchar(fd);
			if (z == 5 || z == 6) {
				poleUsed = z - 4;
				serialFlush(fd);
				resetVariables(&centroid, &frisbeeArea, &frisbeeX, &frisbeeY, &changeX, &changeY, &windowWidthChange, &windowHeightChange, &spotWidthI, &spotWidthII, &spotHeightI, &spotHeightII, &decreaseArea, &goingUp, &methodUsed, &lastFrame, &startCapture, &nCount,
					&widthBy2, &heightBy2, &prevWidthBy2, &prevHeightBy2);
				goto START;
			}
			serialFlush(fd);
		}
	}

	return 0;
}

