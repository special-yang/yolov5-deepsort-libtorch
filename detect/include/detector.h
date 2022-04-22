#ifndef DETECTOR_H
#define DETECTOR_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include<torch/script.h>
#include <torch/torch.h>
#include <string>
#define BUFFSIZE 512




class Detector{

public:
        explicit Detector();
        //std::vector<torch::Tensor> detect(cv::Mat input,torch::Device device);
        std::vector<torch::Tensor> detect(torch::Tensor imgTensor);
        void visualize(cv::Mat & frame, std::vector<torch::Tensor>  &dets);
        // static const int iou_flag = 1; //0 iou-nmx, 1 diou-nms
        std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds);


private:
        torch::jit::script::Module module;
        static const int iou_flag = 1; //0 iou-nmx, 1 diou-nms
        float score_thresh= 0.15;
        float iou_thresh=0.6;
};



#endif //DETECTOR_H