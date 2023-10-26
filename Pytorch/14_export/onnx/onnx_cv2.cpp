#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;


void imageDetect(String onnx_path, String image_path) {
	//加载onnx模型
	cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);
	//加载图片
	cv::Mat image = cv::imread(image_path);
	//调整图片
	cv::Mat blob = cv::dnn::blobFromImage(image,
											1.0 / 255,									//归一化
											cv::Size::Size_(640, 640),					//size
											cv::Scalar::Scalar_(0.485, 0.456, 0.406),	//mean
											false,										//swapRB
											false										//crop
											);
	//设置模型输入
	net.setInput(blob);
	//推理输出结果
	cv::Mat predict = net.forward();
	cout << predict.size << endl;	//1 x 25200 x 15
}


int main() {
	String onnx_path = "D:\\ai\\code\\yolov5-master\\opencv\\best.onnx";
	String image_path = "D:\\ai\\code\\yolov5-master\\opencv\\20210726_135155_710.jpg";
	imageDetect(onnx_path, image_path);

	system("pause");
}