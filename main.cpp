#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>
#include <thread>

#include "handKeyPoints.h"

constexpr unsigned short framesPerSequence = 32;
constexpr unsigned short keypoints = 21;

const char* gestureName[] = { "WAWING" , "SCISSORS", "FLIP", "PUSH&PULL", "OPEN&CLOSE" };

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
public:
    Timer() 
    {
        _start = std::chrono::high_resolution_clock::now();
    }
    ~Timer() 
    {
        Stop();
    }
    void Stop()
    {
        auto _end = std::chrono::high_resolution_clock::now();

        auto _duration = _end - _start;
        
        std::cout << "Duration: " << _duration.count() * 0.000000001 << "\n";
    
        return;
    }
};

void predictGesture(cv::Mat (&sequence)[framesPerSequence], torch::jit::script::Module &handModel,
                    torch::jit::script::Module &gestureModel, bool &predicted, std::string& screenMsg)
{
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor inputTensor = torch::zeros({ 1, keypoints * 2, framesPerSequence });
    Timer timer;
    // Extract keypoint location from the frames
    for (int i = 0; i < framesPerSequence; i++)
    {
        std::vector<std::map<float, cv::Point2f>> handKeypoints;
        std::vector<cv::Rect> handrect;
        handKeypoints = pyramidinference(handModel, sequence[i], handrect);
        
        for (int j = 0; j < handKeypoints.size(); j++)
        {
            if (!handKeypoints[j].empty())
            {
                inputTensor[0][2 * j][i] = handKeypoints[j].begin()->second.x;
                inputTensor[0][2 * j + 1][i] = handKeypoints[j].begin()->second.y;    
            }
        }
    }
    inputs.push_back(inputTensor);

    auto output = gestureModel.forward(inputs).toTensor();
    std::cout << output << '\n';
    
    screenMsg = "You performed " + std::string(gestureName[output.argmax(1).item().toInt()]);
    predicted = true;
}

int main()
{
    torch::jit::script::Module handModel = torch::jit::load("hand.pts");
    assert(handModel != nullptr);

    torch::jit::script::Module gestureModel = torch::jit::load("model.pt");
    assert(gestureModel != nullptr);

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Unable to open camera" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::string screenMsg = "Press space bar and perform the gesture. Press esc to close";
    std::string window = "Hand gesture classification";

    while (1)
    {
        cv::Mat frame;
        cv::Mat sequence[framesPerSequence];

        while (1) 
        {   
            cap >> frame;
            putText(frame, screenMsg, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1);
            cv::imshow(window, frame);
            int key = cv::waitKey(30);
            if (key == 27) goto finish;
            else if (key >= 0)
            {
                {
                    Timer timer;
                    for (int i = 0; i < framesPerSequence; i++)
                    {
                        cv::imshow(window, frame);
                        cap >> frame; 
                        key = cv::waitKey(63);
                        if (key == 27) goto finish;
                        else if (key >= 0)
                            break;
                        sequence[i] = frame.clone();
                    }
                }
                bool predicted = false;
                std::thread thrd(std::ref(predictGesture), std::ref(sequence), 
                                std::ref(handModel), std::ref(gestureModel),
                                std::ref(predicted), std::ref(screenMsg));
                
                // Continue with open camera while the gesture is being predicted
                while (1) {
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