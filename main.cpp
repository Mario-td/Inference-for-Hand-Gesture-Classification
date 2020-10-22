#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

#include "handKeyPoints.h"

constexpr unsigned short framesPerSequence = 32;
constexpr unsigned short keypoints = 21;

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


void predictGesture(std::queue<cv::Mat>& sequence, torch::jit::script::Module& handModel, torch::jit::script::Module& gestureModel)
{
    std::vector<torch::jit::IValue> inputs;
    //torch::Tensor inputTensor = torch::zeros({ 1, keypoints * 2, framesPerSequence });
    Timer timer;
    float vec[1][framesPerSequence][keypoints * 2];
    // Extract keypoint location from the frames
    while (!sequence.empty())
    {
        std::vector<std::map<float, cv::Point2f>> handKeypoints;
        std::vector<cv::Rect> handrect;
        handKeypoints = pyramidinference(handModel, sequence.front(), handrect);
        
        for (int i = 0; i < handKeypoints.size(); i++)
        {
            if (handKeypoints[i].empty())
            {
                vec[0][framesPerSequence - sequence.size()][2 * i] = 0.0;
                vec[0][framesPerSequence - sequence.size()][2 * i + 1] = 0.0;
            }
            else
            {
                vec[0][framesPerSequence - sequence.size()][2 * i] = handKeypoints[i].begin()->second.x;
                vec[0][framesPerSequence - sequence.size()][2 * i + 1] = handKeypoints[i].begin()->second.y;
                //memcpy(inputTensor[0][2*i][framesPerSequence - sequence.size()].data_ptr<float>(), &handKeypoints[i].begin()->second.x, sizeof(float));
                //memcpy(inputTensor[0][2*i+1][framesPerSequence - sequence.size()].data_ptr<float>(), &handKeypoints[i].begin()->second.x, sizeof(float));
            }
        }
        sequence.pop();
    }

    torch::Tensor f = torch::from_blob(std::data(vec), { 1, framesPerSequence, keypoints * 2 });
    std::cout << f[0][0] << "\n";
    f = f.transpose(1, 2);
    std::cout << f[0][0] << "\n";
    inputs.push_back(f);

    auto output = gestureModel.forward(inputs).toTensor();

    std::cout << output << '\n';
    std::cout << "max: " << output.argmax(1).item().toInt() << '\n';
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

    cv::Mat frame;
    
    std::queue<cv::Mat> sequence;

    while (1) {
        cap >> frame;
        cv::imshow("Hand gestures", frame);

        if (cv::waitKey(30) >= 0)
        {   
            Timer timer;
            for (int i = 0; i < framesPerSequence; i++)
            {
                cv::imshow("Hand gestures", frame);
                cap >> frame;
                if (cv::waitKey(63) >= 0)
                    break;
                sequence.push(frame.clone());
            }
            break;
        }
    }
    cv::destroyWindow("Hand gestures");

    predictGesture(sequence, handModel, gestureModel);

	return 0;
}