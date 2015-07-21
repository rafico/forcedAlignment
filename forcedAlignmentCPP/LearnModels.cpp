#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include "PedroFeatures.h"
#include "LearnModels.h"

using namespace boost::filesystem;
using namespace cv;
using namespace std;

LearnModels::LearnModels(const string& models_path, const string& images_path)
{
	auto models_directory = models_path.empty() ? "D:/Dropbox/Irina, Rafi and Yossi/datasets/saintgalldb-v1.0/ground_truth/character_location" : models_path;
	auto images_directory = images_path.empty() ? "D:/Dropbox/Irina, Rafi and Yossi/datasets/saintgalldb-v1.0/data/page_images/" : images_path;

	path dir_path(models_directory, native);
	if (!exists(dir_path))
	{
		std::cerr << "Error: Unable to read models from " << models_directory << std::endl;
		return;
	}

	directory_iterator end_itr; // default construction yields past-the-end
	for (directory_iterator itr(dir_path); itr != end_itr; ++itr)
	{
		if (itr->path().extension() == ".csv")
		{
			cout << "Loading " << itr->path().filename() << endl;
			std::ifstream str(itr->path().string());
			if (!str.good())
			{
				std::cerr << "Error: Unable to read models from " << models_directory << std::endl;
			}

			string fileName = itr->path().filename().stem().string();
			string fullFileName = images_directory + fileName + ".jpg";
			cv::Mat image = cv::imread(fullFileName);

			if (!image.data)
			{
				cerr << "Could not open or find the image " << fullFileName << std::endl;
				return;
			}

			m_trainingImages.insert({ fileName, image });

			std::string	line;
			while (std::getline(str, line))
			{
				ModelInstance mi(fileName, line);
				if ('\0' == mi.m_asciiCode)
				{
					continue;
				}
				auto &mi_vec = m_modelInstances.find(mi.m_asciiCode);
				if (mi_vec != m_modelInstances.end())
				{
					mi_vec->second.push_back(mi);
				}
				else
				{
					std::vector<ModelInstance> vec = { mi };
					m_modelInstances.insert({ mi.m_asciiCode, vec });
				}
			}
		}
	}
}

LearnModels::~LearnModels()
{}


/*

code for prediction using linear SVM.

Mat temp = Mat::zeros(features.rows, 2, CV_64F);
// testing the prediction.
for (int i = 0; i < features.rows; ++i)
{
Mat result;
svm->predict(features.row(i), result);

Mat SV = svm->getSupportVectors();

Mat alpha; Mat svidx;
double rho = svm->getDecisionFunction(0, alpha, svidx);

double dist = SV.dot(features.row(i)) - rho;
temp.at<double>(i, 0) = result.at<float>(0);
temp.at<double>(i, 1) = dist;
}

/*
vector<uchar> tooFewSamples = { 'B', 'D', 'F', 'L', 'P', 'X' };
if (binary_search(tooFewSamples.begin(), tooFewSamples.end(), asciiCode))
{
continue;
}

*/

/* Threshold  Ascii_code X  Y W H */
ModelInstance::ModelInstance(const std::string &fileName, const std::string &csv_line)
	: m_fileName(fileName)
{
	std::stringstream   lineStream(csv_line);
	std::string         cell;

	std::getline(lineStream, cell, ',');
	m_threshold = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_asciiCode = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.x = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.y = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.width = std::stoi(cell) + 1;

	std::getline(lineStream, cell, ',');
	m_window.height = std::stoi(cell) + 1;
}

void LearnModels::samplePos(Mat & posLst, const cv::Size & size, const vector<ModelInstance>& miVec)
{
	uint pxbin = m_params.m_sbin;
	uint position = 0;

	for (const auto& mi : miVec)
	{
		Mat image = m_trainingImages.at(mi.m_fileName);

		// We expand the query to capture some context
		Rect loc(mi.m_window.x - pxbin, mi.m_window.y - pxbin, mi.m_window.width + pxbin, mi.m_window.height + pxbin);
		uint newH = loc.height;
		uint newW = loc.width;

		uint res;
		while (res = newH % pxbin)
		{
			newH -= res;
			loc.y += int(floor(double(res) / 2));
			loc.height -= int(ceil(double(res) / 2));
		}
		while (res = newW % pxbin)
		{
			newW -= res;
			loc.x += int(floor(double(res) / 2));
			loc.width -= int(ceil(double(res) / 2));
		}

		for (auto dx = m_params.m_rangeX.start; dx < m_params.m_rangeX.end; dx += m_params.m_stepSize4PositiveExamples)	{
			for (auto dy = m_params.m_rangeY.start; dy < m_params.m_rangeY.end; dy += m_params.m_stepSize4PositiveExamples)	{
				// Extract image patch
				auto x1 = max(loc.x + dx, 0);
				auto x2 = min(loc.x + loc.width + dx, image.cols - 1);
				auto y1 = max(loc.y + dy, 0);
				auto y2 = min(loc.y + loc.height + dy, image.rows - 1);
				Mat im = image(Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));

				Mat posPatch;
				resize(im, posPatch, size);
				
				Mat feat = PedroFeatures::process(posPatch, m_params.m_sbin);
				feat = feat.t();
				feat.convertTo(feat, CV_32F);
				feat.reshape(1, 1).copyTo(posLst.row(position++));
			}
		}
	}
}


Size LearnModels::computeAverageWindowSize4Char(const std::vector<ModelInstance>& mi_vec)
{
	double accHeight = 0;
	double accWidth = 0;
	for (const auto& instance : mi_vec)
	{
		accHeight += instance.m_window.height - 1;
		accWidth += instance.m_window.width - 1;
	}

	uint avgWidth = static_cast<uint>(round((accWidth / mi_vec.size()) / m_params.m_sbin))*m_params.m_sbin;
	uint avgHeight = static_cast<uint>(round((accHeight / mi_vec.size()) / m_params.m_sbin))*m_params.m_sbin;

	return Size(avgWidth, avgHeight);
}

/* currently implementing 1 vs all, need to try pairs*/
void LearnModels::train()
{
	Mat trainingNeg = imread("D:/Dropbox/Code/test/trainNeg.jpg");
	int h, w;
	Mat negHogFeatures = PedroFeatures::process(trainingNeg, m_params.m_sbin, &h, &w);
	negHogFeatures.convertTo(negHogFeatures, CV_32F);

	for (const auto& modelInstances : m_modelInstances)
	{
		uchar asciiCode = modelInstances.first;

		// Load the trained SVM.
		Ptr<ml::SVM> svm = ml::SVM::create();
		string fileName = m_params.m_svmModelsLocation + to_string(asciiCode) + ".yml";
		path dir_path(fileName, native);
		if (exists(dir_path))
		{
			svm = ml::StatModel::load<ml::SVM>(fileName);
			std::clog << "Loaded svm model for " << asciiCode << " from file." << endl;
		}
		else
		{
			Size windowSz = computeAverageWindowSize4Char(modelInstances.second);

			int modelH = windowSz.height / m_params.m_sbin;
			int modelW = windowSz.width / m_params.m_sbin;

			windowSz.width += 2 * m_params.m_sbin;
			windowSz.height += 2 * m_params.m_sbin;

			uint hogWindowDim = modelH * modelW * 31;
			size_t length = (m_params.m_propNWords + 1)*m_params.m_numTrWords*modelInstances.second.size();
			Mat features(int(length), hogWindowDim, CV_32F);

			size_t numPos = m_params.m_numTrWords*modelInstances.second.size();
			size_t numNeg = m_params.m_propNWords*numPos;

			std::clog << "Loading " << numPos << " positive examples : " << asciiCode << endl;
			samplePos(features, windowSz, modelInstances.second);
			std::clog << "Loading " << numNeg << " Negative examples : " << asciiCode << endl;
			sampleNeg(features, numPos, negHogFeatures, Size(w, h), Size(modelW, modelH), numNeg);

			// Apply L2 - norm.
			NormalizeFeatures(features);

			vector<int> labels;
			labels.assign(numPos, +1);
			labels.insert(labels.end(), numNeg, -1);

			trainSvm(svm, features, labels, asciiCode);
		}
		m_SVMModels.insert({ asciiCode, svm });
	}
}

void LearnModels::trainSvm(Ptr<ml::SVM> svm, const Mat& features, const vector<int> & labels, uchar asciiCode)
{
	Ptr<ml::TrainData> trainData = cv::ml::TrainData::create(features, ml::ROW_SAMPLE, Mat(labels));

	std::clog << "Start training...";
	svm->setKernel(ml::SVM::LINEAR);
	svm->trainAuto(trainData);
	std::clog << "...[done]" << endl;

	string fileName = m_params.m_svmModelsLocation + to_string(asciiCode) + ".yml";
	svm->save(fileName);
}

void LearnModels::getSvmDetector(const Ptr<ml::SVM>& svm, vector<float> & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void LearnModels::drawLocations(Mat &img, const vector<Rect> &locations, const Scalar &color)
{
	for (auto loc : locations)
	{
		rectangle(img, loc, color, 2);
	}
}

void LearnModels::sampleNeg(cv::Mat& features, size_t startPos, const Mat & negHogFeatures, cv::Size imageHogSz, cv::Size modelHogSz, size_t numOfExamples)
{
	// Pick a random starting cell
	random_device rd;
	mt19937 gen(rd());

	uniform_int_distribution<> byDis(0, imageHogSz.height - modelHogSz.height - 1);
	uniform_int_distribution<> bxDis(0, imageHogSz.width - modelHogSz.width - 1);


	for (; startPos < features.rows; ++startPos)
	{
		int by = byDis(gen);
		int bx = bxDis(gen);
		for (int tmpby = by, i = 0; tmpby < by + modelHogSz.height; ++tmpby, ++i)
		{
			auto sp = tmpby*imageHogSz.height + bx;
			Mat temp = negHogFeatures(Range::all(), Range(sp, sp + modelHogSz.width)).t();
			temp = temp.reshape(1, 1);
			Mat aux = features.colRange(i*modelHogSz.width * 31, (i + 1)*modelHogSz.width * 31).rowRange(int(startPos), int(startPos + 1));
			temp.copyTo(aux);
		}
	}
}

void LearnModels::NormalizeFeatures(cv::Mat & features)
{
	Mat temp, rp;
	for (int i = 0; i < features.rows; ++i)
	{
		double rowNorm = 1 / norm(features.row(i));
		features.row(i) = features.row(i) * rowNorm;
	}
}