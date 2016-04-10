//
//  bp.h
//  mnsit
//
//  Created by 陆闻韬 on 16/4/6.
//  Copyright © 2016年 tobias_lu. All rights reserved.
//

#ifndef bp_h
#define bp_h


#endif /* bp_h */
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <assert.h>
using namespace std;


#define num_node_input    784 //input layer = m
#define width_image       28
#define height_image      28
#define num_node_hidden   120 //hidden layer = 100
#define num_node_output   10  //output layer = l

#define ita_i2h_bp         0.1//n1
#define ita_h2o_bp         0.08	 //n2 ni:n2 = sqrt(input):sqrt(hidden)
#define tao               0.01 // fot tao
#define alpha_bp             0.9 //momoent
#define mem_t                0.6

#define patterns_train_BP   60000 //训练模式对数(总数)
#define patterns_test_BP    10000 //测试模式对数(总数)
#define iterations_BP       100 //最大训练次数
#define accuracy_rate_BP    0.925 //要求达到的准确率











class bp
{
public:
    bp(){data_label_train =nullptr; data_input_train = nullptr; data_input_test = nullptr; data_label_test =nullptr; ita_h2o = ita_h2o_bp; ita_i2h = ita_i2h_bp;alpha = alpha_bp;};
    ~bp()
    {
        delete []data_input_train;
        delete []data_label_train;
        delete []data_input_test;
        delete [] data_label_test;
        //delete []
    };
    
    void Initial();
    void readModel(const char* file_name);
  //  void trainModel(const char* file_name);
    void train(const char* file_name);
    
    //test_fun
    
    
private:
    //4 data arrays
    int* data_input_train;
    int* data_input_test;
    int* data_label_train;
    int* data_label_test;
    
    float ita_i2h;
    float ita_h2o;
    float alpha;
    
    //2 weight arrays
    float weightI2H[num_node_input][num_node_hidden];
    float weightH2O[num_node_hidden][num_node_output];
    
    float weightI2H_m[num_node_input][num_node_hidden];
    float weightH2O_m[num_node_hidden][num_node_output];
    
    //2 threshold arrays
    float thresholdH[num_node_hidden];
    float thresholdO[num_node_output];
    
    float thresholdH_m[num_node_hidden];
    float thresholdO_m[num_node_output];
    
    //delta
    float deltaO[num_node_output];
    float deltaH[num_node_hidden];
    
    //output values
    float output_hidden[num_node_hidden];//Yn
    float output_out[num_node_output];//On
    
    
protected:   //test than change to protected
    
    //weight initials
    
    void initialWeightsWithRandomNumbers();
    void initialWeightsWithZeros();
    
    void initialWeightsMem();
    
    
    //calculation functions
    float activationFunction(float x); //sigmoid function vn
    float differentialActivationFunction(float x);//v'n
    
    //forward process
    void hiddenOutput(const int* data);
    void outputOutput();
    
    //backward process
    void updateOutputLayer();
    void updateOutputLayerWithoutMem();
    
    void calculateOutputError(const int * data);
    
    void updateHiddenLayer(const int * data);
    void updateHiddenLayerWithoutMem(const int* data );
    
    //model
    void saveModel(const char* file_name);
    
    //train functions
    float test();   //return accuracy
    
};

int myRand(int i)
{
    return rand()%i;
}

static int reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


static void load_mnist_images(string file_name, int* data)
{
    ifstream file(file_name, std::ios::binary);
    assert(file.is_open());
    
    int magic_num = 0;
    int num_of_images = 0;
    int num_of_rows = 0;
    int num_of_columns = 0;
    
    file.read((char*)&magic_num, sizeof(magic_num));
    magic_num = reverseInt(magic_num);
    file.read((char*)&num_of_images, sizeof(num_of_images));
    num_of_images = reverseInt(num_of_images);
    file.read((char*)&num_of_rows, sizeof(num_of_rows));
    num_of_rows = reverseInt(num_of_rows);
    file.read((char*)&num_of_columns, sizeof(num_of_columns));
    num_of_columns = reverseInt(num_of_columns);
    
    
    //test
    //cout<<magic_num<<" "<<num_of_images<<" "<<num_of_rows<<" "<<num_of_columns<<endl;
    //file.close();
    //
    
    for(int i = 0;i < num_of_images;i++)
    {
        for(int j = 0;j<num_of_rows;j++)
        {
            for (int q = 0;q<num_of_columns;q++)
            {
                unsigned char cur = 0; //read by binary
                file.read((char*)&cur, sizeof(cur));
               
                /*
                if (cur > 128)
                {
                    data[i * num_node_input + j * num_of_rows + q] = 1;
                    
                }
                else
                {
                    data[i * num_node_input + j * num_of_rows + q] = 0;
                
                }
                 */
                data[i * num_node_input + j * num_of_rows + q] = cur - 128;
            }
        }
    }
    
    file.close();
    
    
}

static void load_mnist_labels(string file_name, int* data)
{
    ifstream file(file_name, std::ios::binary);
    assert(file.is_open());
    
    
    int magic_num = 0;
    int num_of_items = 0;
    
    file.read((char*)&magic_num, sizeof(magic_num));
    magic_num = reverseInt(magic_num);
    file.read((char*)&num_of_items, sizeof(num_of_items));
    num_of_items = reverseInt(num_of_items);
    
    //test
    //cout<<magic_num<<" "<<num_of_items<<endl;
    //file.close();
    
    for (int i = 0; i < num_of_items; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        data[i * num_node_output + temp] = 1;
    }
    file.close();
}


void bp::Initial()
{
    //step 1: memset for 4 arrays.
    
    data_input_train = new int[num_node_input * patterns_train_BP];
    memset(data_input_train, 0, sizeof(int)*num_node_input*patterns_test_BP);
    
    data_label_train = new int[num_node_output * patterns_train_BP];
    memset(data_label_train,0, sizeof(int)*num_node_output*patterns_train_BP);
    
    data_input_test = new int[num_node_input * patterns_test_BP];
    memset(data_input_test, 0, sizeof(int)*num_node_input*patterns_test_BP);
    
    data_label_test = new int[num_node_output * patterns_test_BP];
    memset(data_label_test,0, sizeof(int)*num_node_output * patterns_test_BP);
    
    //step 2: initial weights
    
    initialWeightsWithRandomNumbers();
    //initialWeightsWithZeros(); //alternative for 0
    
    //setp 3: load data
    
    string fileInputTrain = "/Users/Tobias_Lu/Documents/data/mnist/train-images.idx3-ubyte";
    string fileLabelTrain = "/Users/Tobias_Lu/Documents/data/mnist/train-labels.idx1-ubyte";
    string fileInputTest = "/Users/Tobias_Lu/Documents/data/mnist/t10k-images.idx3-ubyte";
    string fileLabelTest ="/Users/Tobias_Lu/Documents/data/mnist/t10k-labels.idx1-ubyte";
    
    load_mnist_images(fileInputTrain, data_input_train);
    load_mnist_labels(fileLabelTrain, data_label_train);
    load_mnist_images(fileInputTest, data_input_test);
    load_mnist_labels(fileLabelTest, data_label_test);
    
    cout<<"Initialized!"<<endl;
    //cin.get();
}


void bp::initialWeightsWithRandomNumbers()
{
    srand((unsigned)time(NULL));
    
    for (int i = 0;i < num_node_input;i++)
    {
        for(int j = 0 ; j < num_node_hidden;j++ )
        {
            weightI2H[i][j] = -1 + 2 * ((float)rand())/RAND_MAX;
            weightI2H_m[i][j] = 0;// -1 to 1;
            //cout<<weightI2H[i][j];
        }
    }
    
    
    for (int i = 0;i < num_node_hidden;i++)
    {
        for(int j = 0 ; j < num_node_output;j++ )
        {
            weightI2H[i][j] = -1 + 2 * ((float)rand())/RAND_MAX; // -1 to 1;
            weightH2O_m[i][j] = 0;
            //cout<<weightI2H[i][j];
        }
    }
    
    for (int i = 0; i < num_node_hidden; i++) {
        thresholdH[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        thresholdH_m[i] = 0.0;
    }
    
    for (int i = 0; i < num_node_output; i++) {
        thresholdO[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        thresholdO_m[i] = 0.0;
    }
    
    
    

}

void bp::initialWeightsMem()
{
    for (int i = 0; i<num_node_input;i++)
    {
        for (int j = 0 ; j<num_node_hidden;j++)
        {
        weightI2H_m[i][j] = weightI2H[i][j];
    
        }
        
    }
    
    for (int i = 0; i<num_node_hidden;i++)
    {
        for (int j = 0 ; j<num_node_output;j++)
        {
            weightH2O_m[i][j] = weightH2O[i][j];
            
        }
    }
    
    for (int i = 0; i< num_node_hidden;i++)
    {
        thresholdH_m[i] = thresholdH[i];
    }
    
    for (int i = 0; i< num_node_output;i++)
    {
        thresholdO_m[i] = thresholdO[i];
    }
    
}

void bp::initialWeightsWithZeros()
{
    srand((unsigned)time(NULL));
    for (int i = 0;i < num_node_input;i++)
    {
        for(int j = 0 ; j < num_node_hidden;j++ )
        {
            weightI2H[i][j] = 0;
            weightI2H_m[i][j] =0;
            //cout<<weightI2H[i][j];
        }
    }
    
    for (int i = 0;i < num_node_hidden;i++)
    {
        for(int j = 0 ; j < num_node_output;j++ )
        {
            weightH2O[i][j] = 0;
            weightH2O_m[i][j] = 0;
            //cout<<weightI2H[i][j];
        }
    }
    
    for (int i = 0; i < num_node_hidden; i++) {
        thresholdH[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        thresholdH_m[i] = thresholdH[i];
    }
    
    for (int i = 0; i < num_node_output; i++) {
        thresholdO[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
        thresholdO_m[i] = thresholdO[i];
    }
}


//sigmoid functions

float bp::activationFunction(float x)
{
    return (1.0/( 1.0 + exp(-x)));
}

float bp::differentialActivationFunction(float x)
{
    return (activationFunction(x) * (1.0 - activationFunction(x)));
}

/*
 
//tanh functions
float bp::activationFunction(float x)
{
    return (1.7159*tanh(2*x/3));
}


float bp::differentialActivationFunction(float x)
{
    return (1.7159 * 2 / 3 * (1 - tanh(2 * x / 3) * tanh(2 * x / 3)));
            
}
  */
 
 
 
 



void bp::hiddenOutput(const int *data)
{
    for (int i = 0;i < num_node_hidden; i++) //every hidden layer node
    {
        float temp = 0.0;                          //value for sum of Wt*Xn
        for (int j = 0 ; j < num_node_input;j++) // calculate wt * xn
        {
            temp += weightI2H[j][i] * data[j];
        }
        temp = temp - thresholdH[i];
        output_hidden[i] = activationFunction(temp);
    }
}


void bp::outputOutput()
{
    for (int i = 0 ; i < num_node_output; i++) // every output layer node
    {
        float temp = 0.0;
        for (int j = 0; j < num_node_hidden; j++)
        {
            temp += weightH2O[j][i] * output_hidden[j];
        }
        temp = temp - thresholdO[i];
        output_out[i] = activationFunction(temp);
    }
}

void bp::calculateOutputError(const int *data)
{
    for (int i = 0; i < num_node_output;i++) //every output node
    {
        //float delta_temp;
        deltaO[i]  = (data[i] - output_out[i]) * differentialActivationFunction(output_out[i]); //delta grad
        
        
    }
}


void bp::updateOutputLayer() //data for labels
{
    
    
    for (int i = 0; i< num_node_output;i++)//every hidden node
    {
        for (int j = 0 ; j < num_node_hidden;j++)//update every w[j][i]
        {
            float temp = alpha * weightH2O_m[j][i] + ita_h2o * deltaO[i] * output_hidden[j];
            
            weightH2O[j][i] = weightH2O[j][i] + temp ;
            weightH2O_m[j][i] = temp;
            
            
        }
        float temp = alpha * thresholdO_m[i] + ita_h2o * deltaO[i];
        thresholdO[i] = thresholdO[i] + temp;
        thresholdO_m[i] = temp;
    }
}

void bp::updateOutputLayerWithoutMem()
{
    for (int i = 0; i< num_node_output;i++)//every hidden node
    {
        for (int j = 0 ; j < num_node_hidden;j++)//update every w[j][i]
        {
            //float temp = weightH2O[j][i];
            
            weightH2O[j][i] = weightH2O[j][i]  + ita_h2o * deltaO[i] * output_hidden[j];
           // weightH2O_m[j][i] = temp;
            
            
        }
        //float temp = thresholdO[i];
        thresholdO[i] = thresholdO[i] +  ita_h2o * deltaO[i];
       // thresholdO_m[i] = temp;
    }
}






void bp::updateHiddenLayer(const int* data)//data for input
{
    for (int i = 0 ; i< num_node_hidden;i++)
    {
        float error_temp = 0.0;
        for (int j = 0 ; j< num_node_output; j++)
        {
            error_temp += weightH2O[i][j] * deltaO[j];
        }
        deltaH[i] = error_temp * differentialActivationFunction(output_hidden[i]);
    }
    
    for (int i = 0; i < num_node_hidden; i++)
    {
        for (int j = 0; j < num_node_input;j++)
        {
            float temp = alpha * weightI2H_m[j][i]+ita_i2h * deltaH[i] * data[j];
            
            weightI2H[j][i] = weightI2H[j][i] + temp ;
           weightI2H_m[j][i] = temp;
        }
        float temp = alpha * thresholdH_m[i] +ita_i2h * deltaH[i];
        thresholdH[i] = thresholdH[i] + temp;
        thresholdH_m[i] = temp;
    }
}

void bp::updateHiddenLayerWithoutMem(const int *data)
{
    for (int i = 0 ; i< num_node_hidden;i++)
    {
        float error_temp = 0.0;
        for (int j = 0 ; j< num_node_output; j++)
        {
            error_temp += weightH2O[i][j] * deltaO[j];
        }
        deltaH[i] = error_temp * differentialActivationFunction(output_hidden[i]);
    }
    
    for (int i = 0; i < num_node_hidden; i++)
    {
        for (int j = 0; j < num_node_input;j++)
        {
            //float temp = weightI2H[j][i];
            weightI2H[j][i] = weightI2H[j][i] +ita_i2h * deltaH[i] * data[j];
           // weightI2H_m[j][i] = temp;
        }
       // float temp = thresholdH[i];
        thresholdH[i] = thresholdH[i]  +ita_i2h * deltaH[i];
       // thresholdH_m[i] = temp;
    }
}


void bp::saveModel(const char* file_name)
{
    FILE *f;
    f = fopen(file_name, "wb");
    
    int num_input = num_node_input;
    int num_hidden = num_node_hidden;
    int num_output = num_node_output;
    fwrite(&num_input, sizeof(int),1,f);
    fwrite(&num_hidden, sizeof(int),1,f);
    fwrite(&num_output, sizeof(int), 1, f);
    fwrite(weightI2H, sizeof(weightI2H), 1, f);
    fwrite(thresholdH, sizeof(thresholdH),1,f);
    fwrite(weightH2O, sizeof(weightH2O), 1, f);
    fwrite(thresholdO, sizeof(thresholdO),1,f);
    
    fflush(f);
    fclose(f);
    
    
}

void bp::readModel(const char *file_name)
{
    FILE *f;
    f = fopen(file_name, "rb");
    
    int num_input;
    int num_hidden;
    int num_output;
    fread(&num_input, sizeof(int), 1, f);
    assert(num_input == num_node_input);
    
    fread(&num_hidden, sizeof(int), 1, f);
    assert(num_hidden == num_node_hidden);
    
    fread(&num_output, sizeof(int), 1, f);
    assert(num_output == num_node_output);
    
    fread(weightI2H, sizeof(weightI2H), 1, f);
    fread(thresholdH, sizeof(thresholdH),1,f);
    fread(weightH2O, sizeof(weightH2O), 1, f);
    fread(thresholdO, sizeof(thresholdO),1,f);
    
    fflush(f);
    fclose(f);
}

float backFire(float n ,int t)
{
    return (n * (1 - (t * tao)));
}

void bp::train(const char* file_name)
{
    srand((unsigned)time(NULL));
    
    
    int rounds = 1;
    while (rounds <= iterations_BP)
    {
        int input_pos[patterns_train_BP];
        for (int i =0; i<patterns_train_BP;i++)
        {
            input_pos[i] = i;
        }
        random_shuffle(input_pos, input_pos + patterns_train_BP, myRand);// random input data positions
        cout<<"input sequance shuffled!"<<endl;

        
        cout<<"The "<<rounds<<" round."<<"    ";
        //cin.get();
        
        float accuracy = test();
        
        cout<<"Accuracy: "<<accuracy<<endl;
        if (accuracy >= accuracy_rate_BP)
        {
            saveModel(file_name);
            cout<<"Model trained!"<<endl;
            break;
        }
        
        if (accuracy > mem_t)
        {
       
        
        
            for (int i = 0 ;i < patterns_train_BP; i++)
        
        
            {
            int* data = data_input_train + input_pos[i] * num_node_input; // forward process
            hiddenOutput(data);
            outputOutput();
            
            int* label = data_label_train + input_pos[i] * num_node_output;
            int* data2 = data_input_train + input_pos[i] * num_node_input;
            calculateOutputError(label);
            updateHiddenLayer(data2);
            updateOutputLayer();
            
        
            }
        }
        else
        {
            for (int i = 0 ;i < patterns_train_BP; i++)
                
                
            {
                int* data = data_input_train + input_pos[i] * num_node_input; // forward process
                hiddenOutput(data);
                outputOutput();
                
                int* label = data_label_train + input_pos[i] * num_node_output;
                int* data2 = data_input_train + input_pos[i] * num_node_input;
                calculateOutputError(label);
                updateHiddenLayerWithoutMem(data2);
                updateOutputLayerWithoutMem();
                
                
            }
        }
        
        ita_i2h = backFire(ita_i2h_bp, rounds);
        ita_h2o = backFire(ita_h2o_bp, rounds);
        alpha = backFire(alpha_bp, rounds);
        
        
        
        
        if (rounds == iterations_BP)
        {
            saveModel(file_name);
            cout<<"Model trained!"<<endl;
        }
        cout<<"The "<<rounds<<" ends"<<endl;
        rounds++;
    }
    
    
}

float bp::test()
{
    int count_true = 0;
    
    for(int i = 0;i< patterns_test_BP;i++)
    {
        int* data = data_input_test + i * num_node_input;
        
        
        
        hiddenOutput(data);
        outputOutput();
        
        //cout<<"forward"<<endl;
        
        float  max = -9999;
        int pos = -1;
        
        for(int j = 0; j< num_node_output;j++)
        {
            if (output_out[j] > max)
            {
                max = output_out[j];
                pos = j		 ;
            }
        }
        
        int* label = data_label_test + i * num_node_output;
        if ( label[pos] == 1)
        {
            count_true ++;
        }
        
        sleep(0.001);
        
        
    }
    
    return (count_true * 1.0 / patterns_test_BP);
}










