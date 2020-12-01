#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <thread>

#include "handKeyPoints.h"

// Number of frames per sequence and number of keypoints detected in each frame
constexpr unsigned short framesPerSequence = 32;
constexpr unsigned short keypoints = 21;

// Gesture labels
const char *gestureName[] = {"WAWING", "SCISSORS", "FLIP", "PUSH&PULL", "OPEN&CLOSE"};

// Function for predicting the gesture from the sequence of frames
void predictGesture(cv::Mat (&sequence)[framesPerSequence], torch::jit::script::Module &handModel,
                    torch::jit::script::Module &gestureModel, bool &predicted, std::string &screenMsg)
{
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor inputTensor = torch::zeros({1, keypoints * 2, framesPerSequence});

    // Extracts keypoint location from the frames
    for (int i = 0; i < framesPerSequence; i++)
    {
        std::vector<std::map<float, cv::Point2f>> handKeypoints;
        std::vector<cv::Rect> handrect;
        handKeypoints = pyramidinference(handModel, sequence[i], handrect);

        // Assigns the coordinates of the keypoints to the tensor, for each time step
        for (int j = 0; j < handKeypoints.size(); j++)
            if (!handKeypoints[j].empty())
            {
                inputTensor[0][2 * j][i] = handKeypoints[j].begin()->second.x;
                inputTensor[0][2 * j + 1][i] = handKeypoints[j].begin()->second.y;
            }
    }

    // Forwards the input throught the model
    inputs.push_back(inputTensor);
    auto output = gestureModel.forward(inputs).toTensor();

    screenMsg = "You performed " + std::string(gestureName[output.argmax(1).item().toInt()]);
    // Flag to exit the thread
    predicted = true;
}

int main(int argc, char **argv)
{
    // Loads the model for hand keypoints detection and gesture classification
    torch::jit::script::Module handModel = torch::jit::load("../hand.pts", torch::kCUDA);

    torch::jit::script::Module gestureModel = torch::jit::load("../model.pt");

    // Opens the webcam
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Unable to open camera" << std::endl;
        return -1;
    }

    // Sets proper width and height
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::string screenMsg = "Press space bar and perform the gesture. Press esc to close";
    std::string window = "Hand gesture classification";

    // Start point before performing any gesture
    while (1)
    {
        cv::Mat frame;
        cv::Mat sequence[framesPerSequence];

        // Cam starts recording
        while (1)
        {
            cap >> frame;
            putText(frame, screenMsg, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1);
            cv::imshow(window, frame);

            int key = cv::waitKey(30);
            // Finishes the app any time the user presses Esc
            if (key == 27)
                goto finish;
            // Starts storing the frames of the gesture after pressing any other button
            else if (key >= 0)
            {
                for (int i = 0; i < framesPerSequence; i++)
                {
                    cv::imshow(window, frame);
                    cap >> frame;
                    key = cv::waitKey(63);
                    if (key == 27)
                        goto finish;
                    else if (key >= 0)
                        break;
                    sequence[i] = frame.clone();
                }

                // Predicts the gesture in a thread
                bool predicted = false;
                std::thread thrd(std::ref(predictGesture), std::ref(sequence),
                                 std::ref(handModel), std::ref(gestureModel),
                                 std::ref(predicted), std::ref(screenMsg));

                // Continues with open camera while the gesture is being predicted
                while (1)
                {
                    cap >> frame;
                    putText(frame, "Predicting the gesture...", cv::Point(20, 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                    cv::imshow(window, frame);
                    key = cv::waitKey(30);
                    if (key == 27)
                    {
                        thrd.join();
                        goto finish;
                    }
                    else if (key >= 0 || predicted)
                        break;
                }
                thrd.join();
                break;
            }
        }
    }
finish:
    return 0;
}
