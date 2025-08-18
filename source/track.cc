#include "track.h"

// /**
// @brief Convert Vector to Matrix
// @param data
// @return Eigen::Matrix<float, Eigen::Dynamic, 6>
// */
// Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
//     Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
//     for (int i = 0; i < data.size(); ++i) {
//         for (int j = 0; j < data[0].size(); ++j) {
//             matrix(i, j) = data[i][j];
//         }
//     }
//     return matrix;
// }

// Tracking::Tracking(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_,
//                    int delta_t_, std::string asso_func_, float inertia_, bool use_byte_)
//     : oc_sort_tracker(det_thresh_, max_age_, min_hits_, iou_threshold_,
//                       delta_t_, asso_func_, inertia_, use_byte_) {}
Tracking::Tracking() : byte_tracker(10, 30) {
}
Tracking::~Tracking() {
}

void Tracking::run(cv::Mat& frame, std::vector<Detection>& output) {
    // if (output.empty()) {
    //     Eigen::Matrix<float, Eigen::Dynamic, 6> dets;
    //     std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(dets);
    //     return;
    // }
    
    // std::vector<std::vector<float>> data;
    // cv::Rect box;
    // for (int i = 0; i < output.size(); ++i) {
    //     Detection detection = output[i];
    //     box = detection.box;
    //     std::vector<float> row;

    //     row.push_back(box.x);
    //     row.push_back(box.y);
    //     row.push_back(box.x + box.width);
    //     row.push_back(box.y + box.height);
    //     row.push_back(detection.confidence);
    //     row.push_back(detection.class_id);

    //     data.push_back(row);
    //     if (!data.empty()) {
    //         std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));
    //         data.clear();
    //     }
    // }
    // // std::vector<Eigen::RowVectorXf> res = oc_sort_tracker.update(Vector2Matrix(data));

    std::vector<Object> objects;
    for (const auto& detection : output) {
        Object obj;
        obj.rect = detection.box;
        obj.label = detection.class_id;
        obj.prob = detection.confidence;
        objects.push_back(obj);
    }
    
    std::vector<STrack> output_stracks = byte_tracker.update(objects);
    return;
}

void Tracking::draw_tracks(cv::Mat& frame) {
    // for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
    //     Eigen::Matrix<float, 1, 4> d;
    //     d = oc_sort_tracker.trackers.at(i).get_state();
    //     cv::Rect box(d(0), d(1), d(2) - d(0) + 1, d(3) - d(1) + 1);
    //     cv::Scalar color(oc_sort_tracker.trackers.at(i).color[0], oc_sort_tracker.trackers.at(i).color[1], oc_sort_tracker.trackers.at(i).color[2]);
    //     cv::putText(frame, cv::format("ID:%d, Last update: %d", oc_sort_tracker.trackers.at(i).id, oc_sort_tracker.trackers.at(i).time_since_update), cv::Point(d(0), d(1) - 5), 0, 0.5, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    //     cv::rectangle(frame, box, color, 3);
    // }
    std::vector<STrack> tracks = byte_tracker.tracked_stracks;
    for (int i = 0; i < tracks.size(); i++) {
        vector<float> tlwh = tracks[i].tlwh;
        // bool vertical = tlwh[2] / tlwh[3] > 1.6;
        // if (tlwh[2] * tlwh[3] > 20 && !vertical)
        // {
        //     Scalar s = tracker.get_color(output_stracks[i].track_id);
        //     putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
        //             0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        //     rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        // }
        cv::Scalar s = byte_tracker.get_color(tracks[i].track_id);
        putText(frame, format("%d", tracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                0, 0.6, Scalar(0, 0, 255), 4, LINE_AA);
        rectangle(frame, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 4);
    }

    return;
}

std::vector<cv::Rect> Tracking::getPlates() {
    // std::vector<cv::Rect> res;
    // float scale_w = 2.0f;
    // float scale_h = 8.0f;
    // int padding = 25;
    // for (int i = 0; i < oc_sort_tracker.trackers.size(); i++) {
    //     if (!oc_sort_tracker.trackers.at(i).isSent) {
    //         Eigen::Matrix<float, 1, 4> d;
    //         d = oc_sort_tracker.trackers.at(i).get_state();
    //         int old_w = d(2) - d(0) + 1;
    //         int old_h = d(3) - d(1) + 1;
    //         int new_w = old_w / scale_w + padding;
    //         int new_h = old_h / scale_h + padding;
    //         int x = d(0) + (old_w - new_w) / 2;
    //         int y = d(1) + (old_h - new_h) / 2;
    //         cv::Rect box(x, y, new_w, new_h);
    //         res.push_back(box);
    //         oc_sort_tracker.trackers.at(i).isSent = true;
    //     }
    // }
    
    std::vector<cv::Rect> res;
    std::vector<STrack> tracks = byte_tracker.tracked_stracks;
    float scale_w = 2.0f;
    float scale_h = 8.0f;
    int padding = 25;
    for (int i = 0; i < tracks.size(); i++) {
        if (!tracks[i].isSent) {
            float old_w = tracks[i].tlwh[2] - tracks[i].tlwh[0] + 1;
            float old_h = tracks[i].tlwh[3] - tracks[i].tlwh[1] + 1;
            float new_w = old_w / scale_w + padding;
            float new_h = old_h / scale_h + padding;
            float x = tracks[i].tlwh[0] + (old_w - new_w) / 2;
            float y = tracks[i].tlwh[1] + (old_h - new_h) / 2;
            cv::Rect box(x, y, new_w, new_h);
            res.push_back(box);
            tracks[i].isSent = true;
        }
    }
    return res;
}