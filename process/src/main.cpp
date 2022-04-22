#include<iostream>
#include <opencv2/opencv.hpp>
// #include <torch/script.h>
// #include <algorithm>
#include <time.h>
//#include "extra.h"
#include "detector.h"
#include "DeepSORT.h"
#include<string>
#include<vector>
#include <chrono>
#include <unordered_map> 
#include <sstream>
#include "util.h"
#define BUFFSIZE 512

int main(int argc, const char *argv[]) {
//int main(){
    //init 
    cv::Rect top(0,0,640,180);
    cv::Rect bottom(0,204,640,180);
    int down_count =0; // 进入数量
    int up_count =0;   // 离开数量
    std::unordered_map<int,int> down_id;
    std::unordered_map<int,int> up_id;
    // int font_face = cv::FONT_HERSHEY_COMPLEX; 
    // double font_scale = 2;
	// int thickness = 2;
    

    auto input_path = std::string(argv[1]);
    torch::Device device = torch::kCUDA;
    // read video
    cv::VideoCapture cap(input_path);
    //cv:: VideoCapture cap = cv::VideoCapture("/home/yangshaopeng/data/jishu/0500291320076_20200506072000_454.mp4");
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open cv::VideoCapture");
    }
    std::array<int64_t, 2> orig_dim{int64_t(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(cap.get(cv::CAP_PROP_FRAME_WIDTH))};
    
    // model init
    Detector detect;   //因为视频固定大小1920，1080，所以detect的输入固定到了640,384
    //std::array<int64_t, 2> orig_dim{384,640};
    DeepSORT tracker(orig_dim);


    //write video
    cv::VideoWriter out;
    out.open(
		"demo_count.avi", //输出文件名
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		20, // 帧率 (FPS)static_cast<int>(cap.get(cv::CAP_PROP_FPS))
		cv::Size( 640, 360 ), // 单帧图片分辨率为 640x480
		true // 只输入彩色图
	);

    cv::Mat image, img;
    int frame_id =-1;
    while(cap.isOpened())
    {
        
	    cap.read(image);
        frame_id +=1;
        if(frame_id<40)continue;

        if(image.empty())
        {
           std::cout << "Read frame failed!" << std::endl;
           break;
        }
        //cv::Mat image = cv::imread("/home/yangshaopeng/project/count_people/000256.jpg");
        //std::cout << image.rows <<" " << image.cols <<" " << image.channels() << std::endl;
        // process image and set image to GPU
        auto t1 = std::chrono::steady_clock::now();
        cv::resize(image, img, cv::Size(640, 360));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte).to(device);
        imgTensor = imgTensor.permute({2,0,1}).toType(torch::kFloat);
        imgTensor = imgTensor.div(255).unsqueeze(0);
        imgTensor = torch::nn::functional::pad(imgTensor,torch::nn::functional::PadFuncOptions({0,0,12,12}));

        std::vector<torch::Tensor> dets = detect.detect(imgTensor.clone());
        for (int b=0;b<dets.size();b++){
            //if(dets[b].size(0)==0)  continue;
            
            auto trks = tracker.update(dets[b], imgTensor);
            auto t2=std::chrono::steady_clock::now(); 
            double dr_ms=std::chrono::duration<double,std::milli>(t2-t1).count();  
            std::cout<<" "<<std::endl;  
            std::cout<<frame_id<<"  time cost: "<<dr_ms<<"ms"<<std::endl;
            if(dets[b].size(0)>0) cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            for (auto &t:trks) {
                cv::Point cpt;
                cpt.x = t.box.x + cvRound(t.box.width/2.0); 
                cpt.y = t.box.y + cvRound(t.box.height/2.0);
                if(top.contains(cpt)){
                    if(up_id.count(t.id)){
                        up_count++;
                        up_id.erase(t.id);
                    }
                    if(down_id.count(t.id)==0){
                        down_id[t.id]++;
                    }

                }else if(bottom.contains(cpt)){
                    if(up_id.count(t.id)==0){
                        up_id[t.id]++;
                    }
                    if(down_id.count(t.id)){
                        down_count++;
                        down_id.erase(t.id);
                    }
                }
               
                draw_bbox(img, t.box, std::to_string(t.id), color_map(t.id));
                
            }
            
            char buf[BUFFSIZE];
            snprintf(buf, BUFFSIZE, "result/%06d.png", frame_id);
            
            
            std::stringstream str;
            str <<"up:"<<up_count<<" "<<"down:"<<down_count;
            draw_text(img, str.str(), {0, 0, 0}, {img.cols, 0}, true);
            out.write(img);
            cv::imwrite(buf,img);
       
        }
        if (frame_id>3000) break;
        
    }
}