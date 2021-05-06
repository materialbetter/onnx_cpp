#include "detect.h"


Detect::Detect(const wchar_t* path, Ort::Env& env, Ort::SessionOptions& session_options)
	:session(env, path, session_options)
{
	confThreshold = 0.3;
	nmsThreshold = 0.6;
}

Detect::~Detect()
{

}


void Detect::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat & frame)
{
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 1);
}


std::vector<center_coordinates> Detect::run_net( cv::Mat img)
{
	int w = img.cols;
	int h = img.rows;

	cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1 / 255.F, cv::Size(480, 480), cv::Scalar(), true, false); //调整输入网络的图像格式

	cv::Mat flat = inputBlob.reshape(1, inputBlob.total());
	std::vector<float> input_data = inputBlob.isContinuous() ? flat : flat.clone();//输入数据格式转换

	size_t input_tensor_size = 1 * 3 * 480 * 480;
	std::vector<int64_t> input_node_dims = { 1, 3, 480, 480 };//输入图像大小		
	//创建输入tensor
	//auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_tensor_size, input_node_dims.data(), 4);

	////*************************************************************************

	std::vector<const char*> input_node_names = { "input0" };//输入节点名字，不同的模型不同
	std::vector<const char*> output_node_names = { "output0","output1" };//输出节点名字，不同的模型不同

	// 得到模型返回值
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);

	//*************************************************************************
	//对输出数据进行后续处理
	float* arr0 = output_tensors[0].GetTensorMutableData<float>(); //指向置信度tensor的首个元素 长度before_nms
	float* arr1 = output_tensors[1].GetTensorMutableData<float>();//指向位置信息tensor的首个元素 每4个一组 长度before_nms*4

	//还未经NMS处理，共有before_nms个框的信息
	// 置信度
	std::vector<float> values;
	for (int i = 0; i < before_nms; i++)
	{
		values.push_back(arr0[i]);
		
	}

	//位置  此处信息为与输入图像长宽的比例
	std::vector<ratios> outs(before_nms);
	int j = 0;
	for (int i = 0; i < before_nms; i++)
	{
		outs[i].r1 = arr1[j++];
		outs[i].r2 = arr1[j++];
		outs[i].r3 = arr1[j++];
		outs[i].r4 = arr1[j++];
	}

	std::vector<center_coordinates> all_coordinates;
	//置信度
	std::vector<float> confidences;
	//类别
	std::vector<int> classes;
	std::vector<cv::Rect> boxes;
	for (int i = 0; i < before_nms; i++)
	{
		
		float cof = values[i];
		int cls = 0;
		if (cof >= confThreshold)//根据置信度先筛掉一部分
		{
			//恢复到原图的坐标体系
			float center_x = (float)(outs[i].r1 * w);
			float center_y = (float)(outs[i].r2 * h);
			float width = (float)(outs[i].r3 * w);
			float height = (float)(outs[i].r4 * h);
			float left = center_x - width * 0.5;
			float top = center_y - height * 0.5;

			confidences.push_back(cof);
			classes.push_back(cls);
			boxes.push_back(cv::Rect(left, top, width, height));
		}
	}

	//非极大值抑制处理
	std::vector<int> perfect_indx;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, perfect_indx);


	for (int i = 0; i < perfect_indx.size(); i++)
	{
		int idx = perfect_indx[i];
		cv::Rect box = boxes[idx];
		//roi的坐标信息

		float cof = confidences[idx];
		int cls = classes[idx];
		center_coordinates co = { box.x, box.y, box.x + box.width, box.y + box.height,cof,cls };
		all_coordinates.push_back(co);
		//画出边框
		drawPred(confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, img);
	}

	return all_coordinates;
}
