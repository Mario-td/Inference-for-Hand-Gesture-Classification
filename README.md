# Inference for Hand Gesture Classification
The objective is to perform an inference for a hand gesture classification model, also covered [here](https://github.com/Mario-td/Hand-Gesture-Classification-with-Tensorflow-2.0).
This time, the neural network model is trained using Pytorch library and it is saved to use it with the C++ API. Then the application runs with OpenCV, by using a pre-trained model for hand keypoint detection and the model built for time series classification.
After the application starts running, the user has to press the space bar and perform one of the 5 gestures for 2.5-3 seconds. Then the program predicts which gesture was performed.
The hand keypoint detector model "hand.pts" and functions from the "handKeyPoints.h" file are implemented by the authors of a research paper[1]. 

## Gesture classes
* WAVING                        
* SCISSORS
* FLIP
* PUSH&PULL
* OPEN&CLOSE

![Image 1](/images/WAVING.gif) ![Image 2](/images/SCISSORS.gif) ![Image 3](/images/FLIP.gif) ![Image 4](/images/PUSH&PULL.gif) ![Image 5](/images/OPEN&CLOSE.gif)

## Technologies
Project is created with:
* OpenCV 4.2
* Torch (C++ and Python)
* Scikit-Learn
* Pandas
* Numpy


## References
1. Y. Wang, B. Zhang, C. Peng,"SRHandNet: Real-Time 2D Hand Pose EstimationWith Simultaneous Region Localization", IEEE, Transactions on Image Processing,Volume 29, Pages 2977 - 2986, DOI:10.1109/TIP.2019.2955280, 2019.
