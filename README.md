# ABSTRACT

This project involves tracking of flying Frisbee and pole detection for the problem statement of Robocon 2017 through computer vision techniques. The problem statement requires the robot to launch frisbees from the Throwing Area such that they land on the poles in the Game Field. Colour based detection using histogram and its advantages are discussed. For installation and configuration of OS in raspberry pi, OpenCV in visual studio and OpenMP, refer to the attached documentation pdf.

# Algorithm

1. <b>Color Based Detection</b>
    1. <b>Static Method</b><br />
    Frisbee is detected using colour detection algorithm. First, image is converted to HSV format as it is less sensitive to light. Minimum and maximum thresholds are applied on HUE and VALUE plane.  
    2. <b>Dynamic or Adaptive Method</b><br />
    Colour Thresholding method has a very big disadvantage. A very slight variation in light can change threshold value. To overcome this problem, a dynamic approach to find out threshold values using histogram is proposed here. This method can be applied under following assumptions: 
        * Approximate area of object is known.
        * One colour channel is sufficient to distinguish with other colour. For example, white colour detection can be detected only using HUE channel.
        * Maximum and minimum threshold is known.<br /><br />
    Steps to follow:
        * First find out histogram of channel.
        * Start counting points from minimum index until points are smaller than area of object.
        * Apply this minimum and new threshold values to channel.<br /><br />
    The main ADVANTAGES of this method are that it is background invariant and requires only one channel.<br />
    Procedure to detect poles and Frisbee using this method:
        * Pole Detection:<br />
        Poles are situated at known distances from camera. Their areas are fixed. Also, white colour poles detection require only HUE channel (in HSV format). Minimum and maximum thresholds are taken 0 and 150. A certain portion of poles have arena in background. So, algorithm is applied only to that portion.
        * Frisbee Detection:<br />
        Frisbee Colour is black. This is detected using VALUE plane. Although it sometimes requires two channel which can be converted to one channel by applying a fixed threshold on other channel.
        
2. <b>Noise Filter</b>
To remove small size noises, we used median filter which takes median of aperture around a particular pixel of size given by user and assigns it to that pixel. We choose size of aperture to be “17, 7”.

3. <b>Frisbee Extraction</b>
Eccentricity, angle and area are used to distinguish the Frisbee from other objects. Since the Frisbee shape is like elliptical its eccentricity should be less than 1. A minimum threshold of 0.85 is applied on blobs. Also Frisbee was thrown vertically, its angle should be near to 0 or 180 degree. And then the blob which have closest area to the Frisbee of previous frame may be Frisbee.<br /><br />
But this result may be wrong due to following reasons: \(i) If the Frisbee is in black background. \(ii) Frisbee is not detected due to light illumination.<br /><br />
To check it is Frisbee or not, error between estimated position and predicted position (discussed below) of Frisbee is used. Now, if it comes out that the detected blob doesn’t satisfy aforementioned condition, predicted position will be considered as Frisbee centroid.

4. <b>Centroid Prediction</b>
Centroid is predicted using median of last four changes in coordinates of the Frisbee.

5. <b>Multiprocessing in Raspberry Pi</b>
For multiprocessing, OpenMP library is used. Raspberry Pi 3 has four cores, we use first core to take images and store them in memory to increase frame rate and all other three cores were used to do processing on images. Once, all frames are captured, all cores are used to process image.

6. <b>Communication With Arduino</b>
The communication protocol used to communicate Arduino and Pi is UART. I2C communication protocol can’t be used as same protocol is used by Pi Cam. Also, on using SPI protocol USB remote stops working.

# Important Functions

There are three functions defined in FDetectionFunction.h file which are building blocks to implement above algorithm. These functions are useful in any object recognition problem. A brief description is given below:
1. <b>medFilt2():</b> This function is used to filter the noise using median of mask around a pixel. This function is defined in OpenCV library but it filters the image only using square shaped mask. But in our situation, we needed to use a rectangular mask as Frisbee was elliptical. This function can be implemented by counting number of zeroes in the mask around a pixel. If number of zeroes is less than number of pixel in mask than a value 1 will be assigned to this pixel else 0. 

```c++
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
```
2. <b>mouseHandler():</b> This function is used to debug the code and analysis of image. It prints HSV and RGB values of a pixel when clicking on displayed image. This function is not directly called. It needs to be passed on an inbuilt function setMouseCallback which takes three input – name of window in which image is displayed, mouseHandler function and address of image. For example, If you want to print values of an image stored in frame Mat object displayed in window say “window”, use the statement:
```c++
setMouseCallback("window", mouseHandler, &frame);
```
3. <b>getBlobProps():</b>	This function is used to find out properties of blobs in binary image like contour areas, centroids, eccentricity, dimension, angle and contour itself. Inputs of this function are binary image and a struct which members are vector. After calling this function, contour properties will be stored in struct object.
```c++
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
```
Note: For more detail, refer to FDetectionFunction.h file.

<b>Future Work:</b>
1. Dynamic method of colour detection can be improved for colours requiring more than two channels. For example, if RGB format is used, then to detect white colour all channels will be required but channel required can be made equal to 1 if HSV format is used.
2. Kalman Filter based tracking can be applied for better results in the centroid prediction.
