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
using namespace std;

static int reverseInt(int i);
//static void load_mnist_images(string file_name, int* data);
//static void load_mnist_labels(string file_name, int* data);





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
    bp b;
    b.Initial();
    b.train("/Users/Tobias_Lu/Documents/data/mnist/model.model");
    
    
    
    
    
    return 0;
}






