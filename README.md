# Brain-Tumor-Detection

Medical imaging refers to a set of techniques that aim
to produce visual representations of internal aspects of the
human body in a noninvasive manner. Medical imaging seeks
to reveal the internal organs masked by the skin and bones so
that they can be analysed for any abnormalitites. The resulting images
can be of high resolution which makes any data collection of such images computationally
expensive to process and use with various machine learning paradigms. Usually the images are downscaled
to a size with reasonable processing times and not too much quality degradation. In this project we demonstrate
the superiority of using a GPU over CPU based implementaion of a Logistic Regression model to segment MRI images
of the brain into normal ones (no brain tumor) and abnormal (with brain tumor). The dataset used was adapted 
from [https://www.kaggle.com/ahmedhamada0/brain-tumor-detection?select=no](). The images we preprocessed to
be of size 128 by 128 resulting in a feature vector with 16 384 dimensions. 

## CPU IMPLEMENTATION
The CPU implementation of the Logistic Regression model relies solely on numpy. The underlying physical hardware for this was as follows
 - Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 
 - RAM 16 GB

## GPU IMPLEMENTATION
The GPU implementation of the Logistic Regression model relies on torch and numpy. The underlying GPU hardware was as follows
 - GeForce GTX 1060 with Max-Q Design/PCIe/SSE2  


## RESULTS
The 2 implementations were bench marked in terms of the computational time taken to complete an update of the weights for a given number of iterations 
using gradient descent. Each experiment was run a number of times and the average result taken. The set used for experimentaion is the training set which consist of
1 982 images each of dimensions (128 by 128) or flattened to be 1 by 16 384 feature vectors. The curves give a visual representation of the results

![Alt text](https://github.com/phantom820/Brain-Tumor-Detection/blob/master/figures/runtime.png)
