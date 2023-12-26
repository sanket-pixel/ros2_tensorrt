#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "inference.hpp"




void printHelp() {
    std::cout << "Usage: ./main [--build_engine] [--inference] [--help]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --build_engine   Build the TensorRT engine" << std::endl;
    std::cout << "  --inference      Perform inference using a pre-built engine" << std::endl;
    std::cout << "  --help           Display this help message" << std::endl;
}
using namespace nvinfer1;
int main(int argc, char* argv[]) {
    
     if (argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        printHelp();
        return 0;
    }

    Params params;
    Inference Inference(params); 
     for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--build_engine") == 0) {
            Inference.build();
        }  else if (std::strcmp(argv[i], "--inference") == 0) {
            Inference.buildFromSerializedEngine();
            Inference.initialize_inference();
            Inference.verbose = true;
            cv::Mat img = cv::imread(params.ioPathsParams.image_path,cv::IMREAD_COLOR);;
            std::vector<Object> objects = Inference.do_inference(img);
    
        }
     }


}