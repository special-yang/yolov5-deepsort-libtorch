// debug by yangshaopeng
#include"detector.h"

Detector:: Detector(){
    
    module = torch::jit::load("/home/yangshaopeng/project/count_cplus/yolov5s.torchscript.pt");
    module.to(torch::kCUDA);    // 这里不知道怎么选GPU，
    //torch::Device(torch::kCUDA, 3)不管用
    module.eval();
}

//Detector::~Detector() = default;


void Detector::visualize(cv::Mat & frame, std::vector<torch::Tensor>  &dets){
    if (dets.size() <=0) return;
    // Visualize result
    for (size_t i=0; i < dets[0].sizes()[0]; ++ i){
        float left = dets[0][i][0].item().toFloat() * frame.cols / 640;
        float top = (dets[0][i][1].item().toFloat()-12) * frame.rows / 360;
               
        float right = dets[0][i][2].item().toFloat() * frame.cols / 640;
               
        float bottom = (dets[0][i][3].item().toFloat()-12) * frame.rows / 360;
        float score = dets[0][i][4].item().toFloat();
        int classID = dets[0][i][5].item().toInt();

		cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);
    }
        
    double start = (double)(clock());
    char buf[BUFFSIZE];
    snprintf(buf, BUFFSIZE, "result/%f.png", start);
    cv::imwrite(buf,frame);
}

std::vector<torch::Tensor> Detector::non_max_suppression(torch::Tensor preds)
{
        
        std::vector<torch::Tensor> output;
        
        for (size_t i=0; i < preds.sizes()[0]; ++i)
        {
            torch::Tensor pred = preds.select(0, i); 
            // Filter by scores
            torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
            if (pred.numel() == 0) {
                output.push_back(torch::zeros({0, 6}).to(pred.device()));
                continue;
            }
            // (center_x, center_y, w, h) to (left, top, right, bottom)
            pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
            pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
            pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
            pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);
            // Computing scores and classes
            std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        
            pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
            pred.select(1, 5) = std::get<1>(max_tuple);
            // select people ignor ather class 0 is people class id    debug by yangshaopeng
            pred = torch::index_select(pred,0, torch::nonzero(std::get<1>(max_tuple) == 0).select(1, 0));
            
            if (pred.numel() == 0) {
                output.push_back(torch::zeros({0, 6}).to(pred.device()));
                continue;
            }
            torch::Tensor  dets = pred.slice(1, 0, 6);
            torch::Tensor keep = torch::empty({dets.sizes()[0]}).to(pred.device());
            //torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
            torch::Tensor x1 = dets.select(1, 0).clone();
            torch::Tensor y1 = dets.select(1, 1).clone();
            torch::Tensor x2 = dets.select(1, 2).clone();
            torch::Tensor y2 = dets.select(1, 3).clone();
            torch::Tensor areas = (x2-x1)*(y2-y1);
            std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
            torch::Tensor v = std::get<0>(indexes_tuple).to(dets.device());
            torch::Tensor indexes = std::get<1>(indexes_tuple).to(dets.device());
            int count = 0;
           
            while (indexes.sizes()[0] > 0)
            {
                auto idx = indexes[0].item().toInt();
                keep[count] = idx;
                count += 1;
                indexes = indexes.slice(0,1,indexes.size(0)).clone();
                torch::Tensor xx1 = x1.index_select(0,indexes);
                torch::Tensor yy1 = y1.index_select(0,indexes);
                torch::Tensor xx2 = x2.index_select(0,indexes);
                torch::Tensor yy2 = y2.index_select(0,indexes);
                switch(iou_flag){
                    case 0 :{
                        xx1 = xx1.clamp(x1[idx].item().toFloat(),INT_MAX*1.0).clone();
                        yy1 = yy1.clamp(y1[idx].item().toFloat(),INT_MAX*1.0).clone();
                        xx2 = xx2.clamp(INT_MIN*1.0,x2[idx].item().toFloat()).clone();
                        yy2 = yy2.clamp(INT_MIN*1.0,y2[idx].item().toFloat()).clone();
                        torch::Tensor w = xx2 - xx1;
                        torch::Tensor h = yy2 - yy1;
                        torch::Tensor inter = w * h;
                        torch::Tensor rem_areas = areas.index_select(0,indexes);
                        torch::Tensor ious = inter *1.0/(rem_areas - inter+ areas[idx]);
                        indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0));
                        break;
                        }
                    case 1 :{
                        torch::Tensor iou_x1 = xx1.clamp(x1[idx].item().toFloat(),INT_MAX*1.0).clone();
                        torch::Tensor iou_y1 = yy1.clamp(y1[idx].item().toFloat(),INT_MAX*1.0).clone();
                        torch::Tensor iou_x2 = xx2.clamp(INT_MIN*1.0,x2[idx].item().toFloat()).clone();
                        torch::Tensor iou_y2 = yy2.clamp(INT_MIN*1.0,y2[idx].item().toFloat()).clone();
                        torch::Tensor w = iou_x2 - iou_x1;
                        torch::Tensor h = iou_y2 - iou_y1;
                        
                        w = w.clamp(0,INT_MAX);
                        h = h.clamp(0,INT_MAX);
                        torch::Tensor inter = w * h;
                        torch::Tensor rem_areas = areas.index_select(0,indexes);
                        // FIlter by DIOU
                        torch::Tensor diou_x1 = xx1.clamp(INT_MIN*1.0,x1[idx].item().toFloat()).clone();
                        torch::Tensor diou_x2 = xx2.clamp(x2[idx].item().toFloat(),INT_MAX*1.0).clone();
                        torch::Tensor diou_y1 = yy1.clamp(INT_MIN*1.0,y1[idx].item().toFloat()).clone();
                        torch::Tensor diou_y2 = yy2.clamp(y2[idx].item().toFloat(),INT_MAX*1.0).clone();
                        torch::Tensor c1_0 = diou_x2-diou_x1;
                        torch::Tensor c1_1 = diou_y2-diou_y1;
                        torch::Tensor c2 = c1_0.mul_(c1_0)+ c1_1.mul_(c1_1)+1e-16;
                        torch::Tensor rho2_0 =(xx1+xx2)-(x1[idx]+x2[idx]) ;
                        torch::Tensor rho2_1 = (y1[idx]+y2[idx])-(yy1+yy2);
                        torch::Tensor rho2 = rho2_0*rho2_0/4 + rho2_1*rho2_1/4;
                    
                        torch::Tensor ious = inter *1.0/(rem_areas - inter+ areas[idx])-rho2/c2;
                        indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0));
                        break;
                        }
                }
       
            }
            keep = keep.toType(torch::kInt64);
            auto keep_dets = torch::index_select(dets, 0, keep.slice(0, 0, count));
            keep_dets.select(1, 0) = keep_dets.select(1, 0).clamp(0,640);
            keep_dets.select(1, 1) = keep_dets.select(1, 1).clamp(0,384);
            keep_dets.select(1, 2) = keep_dets.select(1, 2).clamp(0,640);
            keep_dets.select(1, 3) = keep_dets.select(1, 3).clamp(0,384);
            output.emplace_back(keep_dets);
            
        }
       
        return output;
}

//std::vector<torch::Tensor>  Detector::detect(cv::Mat frame,torch::Device device){
std::vector<torch::Tensor>  Detector::detect(torch::Tensor imgTensor){
    // cv::Mat img;
    // cv::resize(frame, img, cv::Size(640, 360));
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte).to(device);
    // imgTensor = imgTensor.permute({2,0,1}).toType(torch::kFloat);
    // imgTensor = imgTensor.div(255).unsqueeze(0);
    // imgTensor = torch::nn::functional::pad(imgTensor,torch::nn::functional::PadFuncOptions({0,0,12,12}));
    auto preds = module.forward({imgTensor}).toTuple()->elements()[0].toTensor();
    std::vector<torch::Tensor> dets = non_max_suppression(preds);
    return dets;


 }