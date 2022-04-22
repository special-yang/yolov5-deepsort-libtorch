#ifndef EXTRA_H
#define EXTRA_H
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>


// TORCH_MODULE(Net);

class Extractor {
public:
    Extractor();
    //torch::Tensor extract(std::vector<cv::Mat> &image); // return GPUTensor
    torch::Tensor extract(std::vector<torch::Tensor> &image); // return GPUTensor
    //torch::Tensor extract(std::vector<cv::Mat> &image)
    // static void print(std::string str);

private:
    torch::jit::script::Module module;
    torch::Tensor MEAN= torch::tensor({0.485f, 0.456f, 0.406f}).view({1, -1, 1, 1}).to(torch::kCUDA);
    torch::Tensor STD= torch::tensor({0.229f, 0.224f, 0.225f}).view({1, -1, 1, 1}).to(torch::kCUDA);    
};
#endif