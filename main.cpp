//
//  main.cpp
//  mnsit
//
//  Created by 陆闻韬 on 16/4/6.
//  Copyright 
//  2016年 tobias_lu.All rights reserved.
//

#include <iostream>
#include <time.h>
#include <algorithm>
#include "bp.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

static int reverseInt(int i);
//static void load_mnist_images(string file_name, int* data);
//static void load_mnist_labels(string file_name, int* data);
int classification();
int test_classification(const int * data);

cv::Mat process(cv::Mat mat);
void DeleteOneColOfMat(cv::Mat& object, int num);
void DeleteOneRowOfMat(cv::Mat& object, int num);
void calCentroid(uchar* data, int width, int height, float *x, float *y);//mass center
void center_mass(cv::Mat mat, int  &x, int &y);

void windows_fwrite_test(const char* file);



int main(int argc, const char * argv[]) {
	// insert code here...
	//test _reverseint

	//load_mnist_images("/Users/Tobias_Lu/Documents/data/mnist/train-images.idx3-ubyte");
	//load_mnist_labels("/Users/Tobias_Lu/Documents/data/mnist/train-labels.idx1-ubyte");

	// bp b;
	//b.Initial();


	/* srand((unsigned)time(NULL));
	int a[6]{1,2,3,4,5,6};
	random_shuffle(a, a+6, myRand);
	for (int i = 0; i < 6; i++)
	{
	cout<<a[i]<<" ";
	}

	*/



	
	//bp b;
	//b.Initial();
	//b.train("/Users/Tobias_Lu/Documents/data/mnist/model.model");
	//b.train("F:\\cdata\\mnist\\model.model");
	

	//windows_fwrite_test("F:\\cdata\\mnist\\test.model");


	// int* testinput = test_image_input();
	// testinput += 784;
	//test_classification(testinput);

	classification();




	return 0;
}



int test_classification(const int * data)
{
	bp bp3;
	//bp3.readModel("/Users/Tobias_Lu/Documents/data/mnist/model.model");
	bp3.readModel("F:\\cdata\\mnist\\model.model");
	int* data_temp = new int[784];

	for (int i = 0; i < 784; i++)
	{
		data_temp[i] = data[i];
	}
	int ret = bp3.predict(data_temp, 28, 28);
	cout << "result = " << ret << endl;
	return ret;

}

int classification()
{



	bp bp2;
	//bp2.readModel("/Users/Tobias_Lu/Documents/data/mnist/model.model");
	bp2.readModel("F:\\cdata\\mnist\\model.model");
	cout << "Model read!" << endl;
	//cin.get();


	//int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//std::string path_images = "/Users/Tobias_Lu/Documents/data/mnist/test_image3/";

	//string path_images = "F:\\cdata\\mnist\\test_image\\";
	string path_images = "F:\\cdata\\mnist\\test_photo_colour\\";

	int* data_image = new int[width_image * height_image];

	for (int i = 2; i < 3; i++) {
		char ch[15];
		sprintf(ch, "%d", i);
		//  cout<<ch<<endl;
		std::string str;
		str = std::string(ch);
		str += ".png";
		str = path_images + str;
		cout << str << endl;
		cin.get();


		cv::Mat mat = cv::imread(str, 2|4);
		if (!mat.data) {
			std::cout << "read image error" << std::endl;
			return -1;
		}

		if (mat.channels() != 1) {
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
		}

		//if (mat.cols != width_image || mat.rows != height_image) {
		//    cv::resize(mat, mat, cv::Size(width_image, height_image));
		//}
		cv::threshold(mat, mat, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);


		//cv::bitwise_not(mat, mat);
		cv::bitwise_not(mat, mat);
		


		
		//mat = process(mat);
		//
		cv::resize(mat, mat, cv::Size(20, 20), CV_INTER_AREA);
		

		//cv::namedWindow("image_ori", CV_WINDOW_AUTOSIZE);
		//cv::imshow("image_ori", mat);
		//cv::waitKey(0);

		int x = 0;//mass center
		int y = 0;

		center_mass(mat, x, y);
		cout << "Center of mass: " << x << ", " << y << endl;
		//cin.get();

		

		x = 13 - x;
		y = 13 - y;
		
		while (x > 7)
		{
			x--;
		}
		while (y > 7)
		{
			y--;
		}


		
		cv::Mat mat_base(28, 28, CV_8U, cv::Scalar(0));
		cv::Mat imageROI;
		imageROI = mat_base(cv::Rect(x,y, 20, 20));
		mat.copyTo(imageROI);

		//copyMakeBorder(mat, mat, 4, 4, 4, 4, 0, cv::Scalar(255, 255, 255));
		
		cout << mat_base.cols << " " << mat_base.rows << endl;


		cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
		cv::imshow("image", mat_base);
		cv::waitKey(0);
		cv::destroyWindow("image");




		//cin.get();
		memset(data_image, 0, sizeof(int) * (width_image * height_image));





		for (int h = 0; h < mat_base.rows; h++) {
			//  cout<<endl;
			uchar* p = mat_base.ptr<uchar>(h);
			for (int w = 0; w < mat_base.cols; w++) {
				int temp = p[w];
				// cout<<temp<<" ";

				if (temp > 128)
				{
					data_image[h* mat_base.cols + w] = 1;

				}
				//   cout<<data_image[h* mat.cols + w]<<" ";

			}

		}


		int ret = bp2.predict(data_image, mat_base.cols, mat_base.rows);
		std::cout << "correct result: " << i << ",    actual result: " << ret << std::endl;
		cin.get();
	}

	delete[] data_image;

	return 0;
}


cv::Mat process(cv::Mat mat)
{
/*
	for (int j = 0; j< mat.cols; j++)
	{
		int temp = 0;
		uchar* p = mat.ptr<uchar>(j);


		for (int i = 0; i < mat.rows; i++)

		{
			
			int cur = (int)p[i];

			temp += cur;


		}
		if (temp == 0)
		{
			DeleteOneColOfMat(mat, j);
			cout << "delete col" << endl;
		}

	}
*/
	for (int j = 0; j< mat.rows; j++)
	{
		int temp = 0;
		uchar* p = mat.ptr<uchar>(j);


		for (int i = 0; i < mat.cols; i++)

		{

			int cur = (int)p[i];

			temp += cur;


		}
		if (temp == 0)
		{
			DeleteOneRowOfMat(mat, j);
			//cout << "delete row" << endl;
		}

	}

	return mat;
}


void DeleteOneColOfMat(cv::Mat& object, int num) {
	if (num<0 || num >= object.cols) {
		cout << " 列标号不在矩阵正常范围内 " << endl;
	}
	else

	{
		if (num == object.cols - 1)
		{
			object = object.t();
			object.pop_back();
			object = object.t();
		}
		else
		{
			for (int i = num + 1; i<object.cols; i++)
			{
				object.col(i - 1) = object.col(i) + cv::Scalar(0, 0, 0, 0);
			}
			object = object.t();
			object.pop_back();
			object = object.t();
		}
	}
}

void DeleteOneRowOfMat(cv::Mat& object, int num) {
	if (num<0 || num >= object.rows) {
		cout << " 列标号不在矩阵正常范围内 " << endl;
	}
	else

	{
		if (num == object.rows - 1)
		{
			object = object.t();
			object.pop_back();
			object = object.t();
		}
		else
		{
			for (int i = num + 1; i<object.rows; i++)
			{
				object.row(i - 1) = object.row(i) + cv::Scalar(0, 0, 0, 0);
			}
			object = object.t();
			object.pop_back();
			object = object.t();
		}
	}
}



void calCentroid(uchar* data, int width, int height, float *x, float *y)//mass center
{
	int i, j;
	float m00 = 0, m10 = 0, m01 = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			uchar tmp = data[i * width + j];
			m00 += tmp;
			m10 += tmp * j;
			m01 += tmp * i;

		}
	}
	if (m00 != 0)
	{
		*x = m10 / m00;
		*y = m01 / m00;
	}
}


void windows_fwrite_test(const char* file)
{
	FILE * f;
	f = fopen(file, "wb");
	cout << "open" << endl;
	int num_input = num_node_input;
	int num_hidden = num_node_hidden;
	int num_output = num_node_output;

	fwrite(&num_input, sizeof(int), 1, f);
	fwrite(&num_hidden, sizeof(int), 1, f);
	fwrite(&num_output, sizeof(int), 1, f);

	cout << "Write done. " << endl;
	fflush(f);
	fclose(f);
}

void center_mass(cv::Mat mat,int  &x, int &y)
{
	double m00, m01, m10;
	cv::Moments moment;
	moment = cv::moments(mat, 1);
	m00 = moment.m00;
	m01 = moment.m01;
	m10 = moment.m10;

	x = (int)(m10 / m00);
	y = (int)(m01 / m00);

}