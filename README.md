# number-classification-neuralnetwork-BP
caution:

1: training data and test data need to be downloaded at http://yann.lecun.com/exdb/mnist/
2: file path in code need to be changed with your own data path.
3: current version is designed with MAC OS X. Windows/Linux needs a few tiny changes. 

4: designed with 2-layer(add 1 input layer) neural network with BP algorithm.
5: At input layer, numbers of node are 28 * 28 = 784(width * height).
6: At hidden layer, the suitable numbers of node are unknown. (Current use is 100).
7: At output layer, numbers of node are 10 for 10 numbers.
7.5: Tricks: constantly-descending η , momentum , randomshuffled input data, random initialed bias.


*8: training data and test data are stored in Binary and the MSB first. Intel processors users need to convert the integers.

9: current accuracy is 90-91.
10: Need to be done:
         predict function.
         accuracy optimized.
         
