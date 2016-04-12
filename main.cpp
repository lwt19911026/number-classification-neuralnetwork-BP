//
//  main.cpp
//  mnsit
//
//  Created by 陆闻韬 on 16/4/6.
//  Copyright © 2016年 tobias_lu. All rights reserved.
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
void DeleteOneColOfMat(cv::Mat& object,int num);
void DeleteOneRowOfMat(cv::Mat& object,int num);
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
    
   // int* testinput = test_image_input();
   // testinput += 784;
    //test_classification(testinput);
    
    classification();
    
    
    
    
    return 0;
}



int test_classification(const int * data)
{
    bp bp3;
    bp3.readModel("/Users/Tobias_Lu/Documents/data/mnist/model.model");
    
    int* data_temp = new int[784];
    
    for (int i = 0;i < 784 ;i++)
    {
        data_temp[i] = data[i];
    }
    int ret = bp3.predict(data_temp, 28, 28);
    cout<<"result = "<<ret<<endl;
    return ret;
    
}

int classification()
{
   
    
  
    bp bp2;
    bp2.readModel("/Users/Tobias_Lu/Documents/data/mnist/model.model");
    
    
    int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::string path_images = "/Users/Tobias_Lu/Documents/data/mnist/test_image3/";
    
    int* data_image = new int[width_image * height_image];
    
    for (int i = 0; i < 5; i++) {
        char ch[15];
        sprintf(ch, "%d", i);
      //  cout<<ch<<endl;
        std::string str;
        str = std::string(ch);
        str += ".png";
        str = path_images + str;
        
        cv::Mat mat = cv::imread(str, 2 | 4);
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
       
        
        
        mat = process(mat);
        cv::resize(mat, mat, cv::Size(20, 20));
        copyMakeBorder(mat, mat, 4, 4, 4, 4, 0, cv::Scalar(255, 255, 255));
        cv::bitwise_not(mat, mat);
        cout<<mat.cols<<" "<<mat.rows<<endl;
        
        /*
        cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
        cv::imshow("image", mat);
        cv::waitKey(0);
        */
        
        
        
        
        //cin.get();
        memset(data_image, 0, sizeof(int) * (width_image * height_image));
        
        
        
        
        
        for (int h = 0; h < mat.rows; h++) {
          //  cout<<endl;
            uchar* p = mat.ptr<uchar>(h);
            for (int w = 0; w < mat.cols; w++) {
                int temp = p[w];
               // cout<<temp<<" ";
                
                    if (temp > 128)
                    {
                        data_image[h* mat.cols + w] = 1;
                        
                    }
             //   cout<<data_image[h* mat.cols + w]<<" ";
                
            }
            
        }
        
        
        int ret = bp2.predict(data_image, mat.cols, mat.rows);
        std::cout << "correct result: " << i << ",    actual result: " << ret << std::endl;
    }
    
    delete[] data_image;
    
    return 0;  
}


cv::Mat process(cv::Mat mat)
{
    
    for (int j = 0 ; j< mat.cols;j++)
    {
        int temp = 0;
        uchar* p = mat.ptr<uchar>(j);
    
    
        for (int i = 0; i < mat.rows;i++)
    
        {
        
            int cur = p[i];
        
            temp+=cur;
        
   
        }
        if (temp ==0)
        {
            DeleteOneColOfMat(mat, j);
        }
    
    }
    for (int j = 0 ; j< mat.rows;j++)
    {
        int temp = 0;
        uchar* p = mat.ptr<uchar>(j);
        
        
        for (int i = 0; i < mat.cols;i++)
            
        {
            
            int cur = p[i];
            
            temp+=cur;
            
            
        }
        if (temp ==0)
        {
            DeleteOneRowOfMat(mat, j);
        }
        
    }
    
    return mat;
}


void DeleteOneColOfMat(cv::Mat& object,int num) {
    if (num<0 || num>=object.cols) 	{
        cout<<" 列标号不在矩阵正常范围内 "<<endl;
    }
    else
    
    {
        if (num == object.cols-1)
        {
            object = object.t();
            object.pop_back();
            object = object.t();
        }
        else
        {
            for (int i=num+1;i<object.cols;i++)
            {
                object.col(i-1) = object.col(i) + cv::Scalar(0,0,0,0);
            }
            object = object.t();
            object.pop_back();
            object = object.t();
        }
    }
}

void DeleteOneRowOfMat(cv::Mat& object,int num) {
    if (num<0 || num>=object.rows) 	{
        cout<<" 列标号不在矩阵正常范围内 "<<endl;
    }
    else
        
    {
        if (num == object.rows-1)
        {
            object = object.t();
            object.pop_back();
            object = object.t();
        }
        else
        {
            for (int i=num+1;i<object.rows;i++)
            {
                object.row(i-1) = object.row(i) + cv::Scalar(0,0,0,0);
            }
            object = object.t();
            object.pop_back();
            object = object.t();
        }
    }
}





