#include "preprocessor.hpp"


cv::Mat Preprocessor::static_resize(cv::Mat& input_image){
    float r = std::min(_resized_width / (input_image.cols*1.0), _resized_height / (input_image.rows*1.0));
    int unpad_w = r * input_image.cols;
    int unpad_h = r * input_image.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(input_image, re, re.size());
    cv::Mat out(_resized_height, _resized_width, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}






