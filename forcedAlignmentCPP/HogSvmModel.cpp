#include <opencv2/core.hpp>
#include "HogSvmModel.h"
#include <string>

HogSvmModel::HogSvmModel(uchar asciiCode, const string& pathCharModels /* = "" */)
	: m_asciiCode(asciiCode),
	m_fileName(pathCharModels + char(asciiCode) + "_" + std::to_string(asciiCode)+".yml")
{}

void HogSvmModel::save2File()
{
	cv::FileStorage fs(m_fileName, cv::FileStorage::WRITE);
	
	vector<int> helper = {m_newH, m_newW, m_bH, m_bW};

	fs << "sizes" << helper << "weight" << weight << "bias" << m_bias;
}

bool HogSvmModel::loadFromFile()
{
	cv::FileStorage fs(m_fileName, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		return false;
	}

	vector<int> helper(4);

	fs["sizes"] >> helper;
	fs["weight"] >> weight;
	fs["bias"] >> m_bias;

	m_newH = helper[0];
	m_newW = helper[1];
	m_bH = helper[2];
	m_bW = helper[3];

	return true;
}