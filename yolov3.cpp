//******************yolov3 with opencv4.0******************
//    
//*********************************************************


#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <gflags/gflags.h>
// #include <ctime
#include <sys/time.h>


#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;
using namespace dnn;
using namespace std;

// #define DRAW_RECT       // draw rectangles of objects on frame or not
// #define INPUT_IMAGES    // if the images are from a folder instead of a video


//**********************************YOLOV3-PARAMS***************************************
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
std::vector<std::string> classes;    // classes of all objects, like person, car, bicycle, etc.



void detect_generator(const std::string &imgs_path,const std::string &yolov3_model_config_path, 
                  const std::string &yolov3_weights_path, const std::string &class_files_path);

std::vector<String> getOutputsNames(const Net& net);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
void yolov3_detect(cv::Mat &frame, const std::string &class_path, const std::string &model_path, 
                   const std::string &weight_path);

void detect_obj(cv::Mat& frame, const std::vector<cv::Mat>& outs);
int64_t getCurrentTime();



int main(int argc, char** argv)
{
    // path of yolov3 model from models directory
    std::string yolov3_model_config_path = "/home/goldenridge/wzj/tools/video2images_pipeline/models/yolov3.cfg";
    // path of yolov3 weight from models directory
    std::string yolov3_weights_path = "/home/goldenridge/wzj/tools/video2images_pipeline/models/yolov3.weights";
    // path of COCO classes file from the project directory
    std::string class_files_path = "/home/goldenridge/wzj/tools/video2images_pipeline/coco.names";

    // input image path from image_path directory
    std::string img_path(argv[1]);
    // directory where to save the converted images
    // std::string save_dir = " ../v3_detect/";
    
    // call video2images function to convert a video to a set of images,
    // and remove all images that do not contain the specified classes
    detect_generator(img_path, yolov3_model_config_path,
                        yolov3_weights_path, class_files_path);

    return 0;

}



// Get the names of the output layers
std::vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty()){
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    } // end-if
    return names;
} // end-getOutputsNames



// Remove the bounding boxes with low confidence using non-maxima suppression
void detect_obj(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
     
    for (size_t i = 0; i < outs.size(); ++i){
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){

            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold){

                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            } // end-if
        } // end-for-j
    } // end-for-i
     
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i){
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        //draw rectangles on the frame, commented by default
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                  box.x + box.width, box.y + box.height, frame);
    } // end-detect_obj
}



// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame){
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    
    if (!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    } // end-if
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
    cv::imshow("test", frame);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
} // end-drawPred



void yolov3_detect(cv::Mat &frame, const std::string &classesFile, const std::string &modelConfiguration, 
                    const std::string &modelWeights)
{
    // load names of classes
    ifstream ifs(classesFile.c_str());
    std::string line;
    while(getline(ifs, line)){
        classes.push_back(line);
    }

    // load the network
    // Net net = readNetFromTorch(modelConfiguration, modelWeights);
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open the image file
    cv::Mat blob;

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_AUTOSIZE);

    // Create a 4D blob from a frame
    blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    // Sets the input to the network
    net.setInput(blob);

    //Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    detect_obj(frame, outs);
}


int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void detect_generator(const std::string &imgs_path, const std::string &yolov3_model_config_path, 
                        const std::string &yolov3_weights_path, const std::string &class_files_path)
{
    std::string pattern = imgs_path + "/*.jpg"; 

    // 必须cv的String 
    std::vector<cv::String> fn; 

    glob(pattern, fn, false); 
    size_t count = fn.size(); 
    cout << count << endl; 
    for (int i = 0; i < count; i++){ 
        stringstream str;
        //str << "imgs_" << i << ".jpg";
        int index = fn[i].find_last_of("/");
        str << fn[i].substr(index+1);
        // cout << "imgname: " << str.str() << endl;
        cv::Mat frame = cv::imread(fn[i]);
        // cv::resize(frame, frame, Size(1920, 1080), (0, 0), (0, 0), INTER_LINEAR);
        int64_t timestamp_start = getCurrentTime();
        // std::string timestamp_str = to_string(timestamp_int);
        // std::string sec = timestamp_str.substr(0, 9);
        // std::string nsec = timestamp_str.substr(9, 4);
        yolov3_detect(frame, class_files_path, yolov3_model_config_path, yolov3_weights_path);
        int64_t timestamp_end = getCurrentTime();
        std::cout << "Total time: " << timestamp_end - timestamp_start << std::endl;
    } // end-for
}