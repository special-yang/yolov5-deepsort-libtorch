#include "extra.h"
#include <vector>
Extractor::Extractor() {
    module = torch::jit::load("ckpt.pt");
    module.to(torch::kCUDA);
    module.eval();
    
}

// void Extractor::print(std::string str) {
//     std::cout << "utils print:" << str << std::endl;
// }

torch::Tensor Extractor::extract(std::vector<torch::Tensor> &image) {
    if (image.empty()) {
        return torch::empty({0, 512});
    }
    // std::vector<torch::Tensor> resized;
    //  for (auto &x:image) {
    //     cv::resize(x, x, {64, 128});
    //     cv::cvtColor(x, x, cv::COLOR_RGB2BGR);
    //     x.convertTo(x, CV_32F, 1.0 / 255);
    //     resized.push_back(torch::from_blob(x.data, {128, 64, 3}, torch::kByte).to(torch::kCUDA));
    // }
    //auto img_tensor = torch::stack(resized).cuda().permute({0, 3, 1, 2}).sub_(MEAN).div_(STD);
    auto img_tensor = torch::cat(image).sub_(MEAN).div_(STD);
    // std::cout<<img_tensor.sizes()<<" extract tensor shape"<<std::endl;
    // std::cout<<(module.forward({img_tensor}).toTensor()).sizes()<<" extract tensor shape"<<std::endl;
    
    return(module.forward({img_tensor}).toTensor());
    // torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte).to(torch::kCUDA);
    // img_tensor = img_tensor.permute({0, 3, 1, 2});
    // img_tensor = img_tensor.toType(torch::kFloat);
    // img_tensor = img_tensor.div(255);
    // img_tensor = img_tensor.sub_(MEAN).div_(STD);
    // torch::Tensor output = module.forward({img_tensor}).toTensor();
    // return output;  
}