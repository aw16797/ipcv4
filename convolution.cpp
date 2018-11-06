// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <iostream>
#include <cmath>
//use CV_32FC3 for images ie. 32
//convert back to 8U at end of program

using namespace cv;
using namespace std;

void GaussianBlur(
	cv::Mat &input,
	int size,
	cv::Mat &blurredOutput);

void sobelX(
	cv::Mat &output
);

void sobelY(
	cv::Mat &output
);

void sobel(
	cv::Mat &input,
	cv::Mat &OutputX,
	cv::Mat &OutputY,
	cv::Mat &OutputM,
	cv::Mat &OutputD
);

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }
 //******END OF IMAGE READING*********

  //greyscale conversion
	Mat gray_image;
  cvtColor( image, gray_image, CV_BGR2GRAY );
	//
  // //blur gray image
  // Mat blurgray_image;
  // GaussianBlur(gray_image,23,blurgray_image);

  //convert blurred gray image from 8 to 32
  // Mat image32;
  // blurgray_image.convertTo(image32,CV_32F);

	// Mat image32;
  // image.convertTo(image32,CV_32F);

  //define new images to hold convolved image
 	Mat imageX(gray_image.size(), CV_32FC1);
	Mat imageY(gray_image.size(), CV_32FC1);
	Mat imageM(gray_image.size(), CV_32FC1);
	Mat imageD(gray_image.size(), CV_32FC1);

	sobel(gray_image,imageX,imageY,imageM,imageD);

	//thresholding
	Mat imageMT(gray_image.size(), CV_32FC1);
	threshold(imageM,imageMT, 0, 255, THRESH_BINARY);

	imwrite("mt.jpg", imageMT);

	// char* window1 = "Sobel X Derivate";
	// char* window2 = "Sobel Y Derivate";
	// char* window3 = "Sobel Magnitude";
	// char* window4 = "Sobel Direction";
  // namedWindow( window1, CV_WINDOW_AUTOSIZE );
	// namedWindow( window2, CV_WINDOW_AUTOSIZE );
	// namedWindow( window3, CV_WINDOW_AUTOSIZE );
	// namedWindow( window4, CV_WINDOW_AUTOSIZE );
	//
	// Mat imageY8;
	// Mat imageX8;
	// Mat imageM8;
	// Mat imageD8;
	// imageX.convertTo(imageX8,CV_8U);
	// imageY.convertTo(imageY8,CV_8U);
	// imageM.convertTo(imageM8,CV_8U);
	// imageD.convertTo(imageD8,CV_8U);
	//
	// imshow( window1, imageX8 );
	// imshow( window2, imageY8 );
	// imshow( window3, imageM8 );
	// imshow( window4, imageD8 );

  //waitKey(0);

  return 0;
}

void sobelX( cv::Mat &output ) {
		output.create(3,3,CV_32FC1);
		output.at<float>(0,0) = (float) -1;
		output.at<float>(0,1) = (float) 0;
		output.at<float>(0,2) = (float) 1;
		output.at<float>(1,0) = (float) -2;
		output.at<float>(1,1) = (float) 0;
		output.at<float>(1,2) = (float) 2;
		output.at<float>(2,0) = (float) -1;
		output.at<float>(2,1) = (float) 0;
		output.at<float>(2,2) = (float) 1;
}

void sobelY( cv::Mat &output ) {
	 output.create(3,3,CV_32FC1);
	 output.at<float>(0,0) = (float) -1;
	 output.at<float>(0,1) = (float) -2;
	 output.at<float>(0,2) = (float) -1;
	 output.at<float>(1,0) = (float) 0;
	 output.at<float>(1,1) = (float) 0;
	 output.at<float>(1,2) = (float) 0;
	 output.at<float>(2,0) = (float) 1;
	 output.at<float>(2,1) = (float) 2;
	 output.at<float>(2,2) = (float) 1;
}

void sobel(cv::Mat &input, cv::Mat &OutputX, cv::Mat &OutputY, cv::Mat &OutputM, cv::Mat &OutputD)
{
	// intialise the output using the input
	// OutputX.create(input.size(), input.type());
	// OutputY.create(input.size(), input.type());
  Mat kX;
	Mat kY;
  sobelX(kX);
	sobelY(kY);

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

  //create padded input to negate border problems
	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convolution
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			float sumX = 0.0;
			float sumY = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					float kernelvalX = kX.at<float>( kernelx, kernely );
					float kernelvalY = kY.at<float>( kernelx, kernely );

					// do the multiplication
					sumX += imageval * kernelvalX;
					sumY += imageval * kernelvalY;
				}
			}
			// set the output value as the sum of the convolution
			OutputX.at<float>(i, j) = sumX;
			OutputY.at<float>(i, j) = sumY;
			OutputM.at<float>(i, j) = sqrt(pow(sumX,2)+pow(sumY,2));
			OutputD.at<float>(i, j) = atan((sumX)/(sumY));

		}
	}
	imwrite( "xd.jpg", OutputX );
	imwrite( "yd.jpg", OutputY );
	imwrite( "mag.jpg", OutputM );
	imwrite( "dir.jpg", OutputD );
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;



	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
