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
#include <opencv2/opencv.hpp>
using namespace std;

static int reverseInt(int i);
//static void load_mnist_images(string file_name, int* data);
//static void load_mnist_labels(string file_name, int* data);
int classification();




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
   // b.Initial();
   // b.train("/Users/Tobias_Lu/Documents/data/mnist/model.model");
    
    
    classification();
    
    
    
    
    return 0;
}

int classification()
{
   
    
  
    bp bp2;
    bp2.readModel("/Users/Tobias_Lu/Documents/data/mnist/model.model");
    
    
    int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::string path_images = "/Users/Tobias_Lu/Documents/data/mnist/test_image2/";
    
    int* data_image = new int[width_image * height_image];
    
    for (int i = 0; i < 10; i++) {
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
        
        if (mat.channels() == 3) {
            cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
        }
        
        if (mat.cols != width_image || mat.rows != height_image) {
            cv::resize(mat, mat, cv::Size(width_image, height_image));
        }
        
        memset(data_image, 0, sizeof(int) * (width_image * height_image));
        
        for (int h = 0; h < mat.rows; h++) {
            uchar* p = mat.ptr(h);
            for (int w = 0; w < mat.cols; w++) {
                
                    if (p[w] > 128)
                    {
                        data_image[h* mat.cols + w] = 1;
                    }
                
            }
        }
        
        int ret = bp2.predict(data_image, mat.cols, mat.rows);
        std::cout << "correct result: " << i << ",    actual result: " << ret << std::endl;
    }
    
    delete[] data_image;
    
    return 0;  
}






