
//参考 https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
#pragma once
#ifndef ONNX_DETECT_H
#define ONNX_DETECT_H

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <regex>
#include <onnxruntime_cxx_api.h>



struct ratios
{
	float r1;//框信息还未处理时的与长宽比例
	float r2;
	float r3;
	float r4;
};

struct center_coordinates
{
	//roi区域的左上角以及长宽
	float left;
	float top;
	float width;
	float height;

	float conf;//置信度
	int cls;//类别

};

class Detect
{
private:
	// 初始化参数
	float confThreshold; // 置信度阈值
	float nmsThreshold;  // 非极大值的抑制阈值
	const int before_nms = 3375;
	Ort::Session session;
public:
	Detect(const wchar_t* path, Ort::Env& env, Ort::SessionOptions& session_options);
	~Detect();
	std::vector<center_coordinates> run_net(cv::Mat imgs);
	void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat & frame);

	
};

#endif // !ONNX_DETECT_H


