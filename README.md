# Inference for Hand Gesture Classification
The objective is to perform an inference for a hand gesture classification model, also covered [here](https://github.com/Mario-td/Hand-Gesture-Classification-with-Tensorflow-2.0).
This time, the neural network model is trained using Pytorch library and it is saved to use it with the C++ API. Then the application runs on Win10 or Ubuntu 20.04.1 LTS, by using a pre-trained model for hand keypoint detection and the model built for time series classification.
After the application starts running, the user has to press the space bar and perform one of the 5 gestures for 2.5-3 seconds. Then the program predicts which gesture was performed.
The hand keypoint detector model "hand.pts" and functions from the "handKeyPoints.cpp" file are implemented by the authors of a research paper[1]. 

## Build
Make sure to get the compatible NVIDIA drivers, CUDA v>=10.1, cuDNN v>=7.6, OpenCV4, and libtorch.

Clone this repository 
```shell
git clone https://github.com/Mario-td/Inference-for-Hand-Gesture-Classification.git
cd Inference-for-Hand-Gesture-Classification
```

Build the executables
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

Run the application
```shell
./InferenceHandGestures
```

## Gesture classes
* WAVING                        
* SCISSORS
* FLIP
* PUSH&PULL
* OPEN&CLOSE

![Image 1](/images/WAVING.gif) ![Image 2](/images/SCISSORS.gif) ![Image 3](/images/FLIP.gif) ![Image 4](/images/PUSH&PULL.gif) ![Image 5](/images/OPEN&CLOSE.gif)

## Other used libraries
* Scikit-Learn
* Pandas
* Numpy


## References
1. Y. Wang, B. Zhang, C. Peng,"SRHandNet: Real-Time 2D Hand Pose EstimationWith Simultaneous Region Localization", IEEE, Transactions on Image Processing,Volume 29, Pages 2977 - 2986, DOI:10.1109/TIP.2019.2955280, 2019.
