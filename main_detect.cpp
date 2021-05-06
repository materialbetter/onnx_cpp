#include "detect.h"
#include <ctime>
#include <io.h>

#include "cuda_provider_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <regex>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <filesystem>

using namespace std;

void getAllFiles(std::string path, regex flag, vector<string>& files)
{
	//文件句柄  
	long long  hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;  //很少用的文件信息读取结构
	string p;  //string类很有意思的一个赋值函数:assign()，有很多重载版本
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))  //判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					//files.push_back(p.assign(path).append("/").append(fileinfo.name));//保存文件夹名字
					getAllFiles(p.assign(path).append("/").append(fileinfo.name), flag, files);//递归当前文件夹
																							   //getAllFiles(p.assign(path).append("/").append(fileinfo.name), files_landmarks);//递归当前文件夹
				}
			}
			else    //文件处理
			{
				if (regex_search(fileinfo.name, flag))
				{
					files.push_back(p.assign(path).append("/").append(fileinfo.name));//文件名	
					//files.push_back(path+"\\"+fileinfo.name);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);  //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
	sort(files.begin(), files.end());
}
clock_t t1, t2;


int main()
{
	string images_path = "test";
	regex regPLY(".bmp");
	vector<string> vecPLYFileName;
	getAllFiles(images_path, regPLY, vecPLYFileName);
	const wchar_t* detect_model_path = L"model.onnx";  //检测模型路径

	////初始化模型环境
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MobilennxEnv");
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	OrtCUDAProviderOptions cudaD = OrtCUDAProviderOptions{ 0,OrtCudnnConvAlgoSearch::EXHAUSTIVE,std::numeric_limits<size_t>::max(),0,true };
	session_options.AppendExecutionProvider_CUDA(cudaD);

	//构造检测对象
	Detect Detect_brick1(detect_model_path, env, session_options);

	int index = 0;
	for (auto pathn : vecPLYFileName)
	{
		cv::Mat image = cv::imread(pathn);
		t1 = clock();
		std::vector<center_coordinates> sum_center = Detect_brick1.run_net(image);  //返回roi区域的左上角以及长宽  置信度以及类别信息

		t2 = clock();
		std::cout << t2 - t1 << endl;

		std::stringstream ss;
		std::string str;
		ss << index;
		ss >> str;
		cv::imwrite("save/"+str+".bmp",image);
		index++;
	}
}