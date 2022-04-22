#include <algorithm>

#include "DeepSORT.h"
#include "extra.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

struct DeepSORT::TrackData {
    KalmanTracker kalman;
    FeatureBundle feats;
};

DeepSORT::DeepSORT(const array<int64_t, 2> &dim)
        : extractor(make_unique<Extractor>()),
          manager(make_unique<TrackerManager<TrackData>>(data, dim)),
          feat_metric(make_unique<FeatureMetric<TrackData>>(data)) {}


DeepSORT::~DeepSORT() = default;

vector<Track> DeepSORT::update(torch::Tensor &dets, torch::Tensor ori_img) {
   
    
    manager->predict();
    manager->remove_nan();
    if (dets.size(0)>0){
    std::vector<cv::Rect2f> detections;
    std::vector<torch::Tensor> inp_tensor;
    for (int n_box=0;n_box<dets.size(0);n_box++){
            auto x0= dets[n_box][0].item().toFloat(),
                 y0= dets[n_box][1].item().toFloat(),
                 x1= dets[n_box][2].item().toFloat(),
                 y1= dets[n_box][3].item().toFloat();
            detections.push_back(cv::Rect2f(x0,y0,x1-x0,y1-y0));
            torch::Tensor inp = ori_img.slice(2,(int)y0,(int)y1);
            inp = ori_img.slice(3,(int)x0,(int)x1);
            inp = torch::nn::functional::interpolate(inp, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({128,64})).mode(torch::kNearest));
            inp_tensor.emplace_back(inp);
        }
    torch::Tensor det_feature = extractor->extract(inp_tensor);
    

    //跟踪好像只能一帧一帧的来，所以batch_size 默认为1？？？？
    auto matched = manager->update(
            detections,
            [this, &detections,&det_feature](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(data[t].kalman.rect());
                }
                //vector<cv::Mat> boxes;
                vector<torch::Tensor> boxes;
                vector<cv::Rect2f> det;
                //debug by yangshaopeng
                for (auto d:det_ids) {
                    det.push_back(detections[d]);
                    boxes.emplace_back(det_feature[d]);
                    
                }
                auto iou_mat = iou_dist(det, trks);
                //auto feat_mat = feat_metric->distance(extractor->extract(boxes), trk_ids);
                auto feat_mat = feat_metric->distance(torch::stack(boxes), trk_ids);
                feat_mat.masked_fill_((iou_mat > 0.8f).__ior__(feat_mat > 0.2f), INVALID_DIST);
                return feat_mat;
            },
            [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.emplace_back(data[t].kalman.rect());
                }
                vector<cv::Rect2f> det;
                for (auto &d:det_ids) {
                    det.emplace_back(detections[d]);
                }
                auto iou_mat = iou_dist(det, trks);
                iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
                return iou_mat;
            });

    //vector<cv::Mat> boxes;
    vector<torch::Tensor> boxes;
    vector<int> targets;
    //for (auto[x, y]:matched) {
    for (auto xy : matched) {
        int x,y;
        x = std::get<0>(xy);
        y = std::get<1>(xy);
        targets.emplace_back(x);
        boxes.emplace_back(det_feature[y]);
        // boxes.emplace_back(ori_img(detections[y]));
    }
    //std::count<<torch::stack(boxes)<<std::endl;
    feat_metric->update(torch::stack(boxes), targets);

    manager->remove_deleted();
    }
    return manager->visible_tracks();
}
