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

LearnModels::LearnModels()
	: m_numRelevantWordsByClass(UCHAR_MAX+1, 0)
{
	loadTrainingData();
}


void LearnModels::loadTrainingData()
{
	path dir_path(m_params.m_pathDocuments, native);
	if (!exists(dir_path))
	{
		cerr << "Error: Unable to read models from " << m_params.m_pathDocuments << std::endl;
		return;
	}

	uint globalIdx=0;
	directory_iterator end_itr; // default construction yields past-the-end
	for (directory_iterator itr(dir_path); itr != end_itr; ++itr)
	{
		if (itr->path().extension() == ".csv")
		{
			cout << "Loading " << itr->path().filename() << endl;
			std::ifstream str(itr->path().string());
			if (!str.good())
			{
				std::cerr << "Error: Unable to read models from " << itr->path().string() << std::endl;
			}
			
			string fileName = itr->path().filename().stem().string();
			string fullFileName = m_params.m_pathImages + fileName + ".jpg";
			cv::Mat image = cv::imread(fullFileName);

			if (!image.data)
			{
				cerr << "Could not open or find the image " << fullFileName << std::endl;
				return;
			}

			vector<CharInstance> charsVec;
			std::string	line;
			while (std::getline(str, line))
			{
				charsVec.push_back(CharInstance(fullFileName, globalIdx, line));
				auto &ci = charsVec.back();
				if ('\0' == ci.m_asciiCode)
				{
					charsVec.pop_back();
					continue;
				}
				++globalIdx;

				uint classNum;
				auto iter = m_classes.find(ci.m_asciiCode);
				if (iter != m_classes.end())
				{
					classNum = iter->second;
				}
				else
				{
					classNum = uint(m_classes.size());
					m_classes.insert({ ci.m_asciiCode, classNum });
				}
				ci.m_classNum = classNum;
				++m_numRelevantWordsByClass[classNum];
			}
			m_docs.push_back(Doc(image, charsVec.size(), move(charsVec), image.rows, image.cols, fullFileName));
		}
	}
	auto iter = find_if(m_numRelevantWordsByClass.begin(), m_numRelevantWordsByClass.end(), [](uint cnt){return cnt == 0; });
	size_t nonZeroCnt = distance(m_numRelevantWordsByClass.begin(), iter);
	m_numRelevantWordsByClass.resize(nonZeroCnt);
	
	size_t numDocs = m_docs.size();
	m_relevantBoxesByClass.resize(numDocs*nonZeroCnt);
	for (int i = 0; i < numDocs; ++i)
	{
		const auto& chars = m_docs[i].m_chars;
		for (const auto& ch : chars)
		{
			uint classNum = ch.m_classNum;
			Rect loc = ch.m_loc;
			size_t idx = i*nonZeroCnt + classNum;
			m_relevantBoxesByClass[idx].push_back(loc);
		}
	}
}

/*

void LearnModels::samplePos(Mat & posLst, const cv::Size & size, const vector<ModelInstance>& miVec)
{
	uint pxbin = m_params.m_sbin;
	uint position = 0;

	for (const auto& mi : miVec)
	{
		Mat image = m_trainingImages.at(mi.m_pathIm);

		// We expand the query to capture some context
		Rect loc(mi.m_loc.x - pxbin, mi.m_loc.y - pxbin, mi.m_loc.width + pxbin, mi.m_loc.height + pxbin);
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


void LearnModels::computeAverageWindowSize4Char()
{
	for (const auto& modelInstances : m_modelInstances)
	{
		uchar asciiCode = modelInstances.first;
		const auto& miVec = modelInstances.second;

		double accHeight = 0;
		double accWidth = 0;
		for (const auto& instance : miVec)
		{
			accHeight += instance.m_loc.height - 1;
			accWidth += instance.m_loc.width - 1;
		}

		uint avgWidth = static_cast<uint>(round((accWidth / miVec.size()) / m_params.m_sbin))*m_params.m_sbin;
		uint avgHeight = static_cast<uint>(round((accHeight / miVec.size()) / m_params.m_sbin))*m_params.m_sbin;
		m_WindowSz.insert({ asciiCode, Size(avgWidth, avgHeight) });
	}
}

// currently implementing 1 vs all, need to try pairs
void LearnModels::train()
{
	Mat trainingNeg = imread("D:/Dropbox/Code/forcedAlignment/trainNeg.jpg");
	int h, w;
	Mat negHogFeatures = PedroFeatures::process(trainingNeg, m_params.m_sbin, &h, &w);
	negHogFeatures.convertTo(negHogFeatures, CV_32F);

	computeAverageWindowSize4Char();

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
			Size windowSz = m_WindowSz.at(asciiCode);

			int modelH = windowSz.height / m_params.m_sbin;
			int modelW = windowSz.width / m_params.m_sbin;

			windowSz.width += 2 * m_params.m_sbin;
			windowSz.height += 2 * m_params.m_sbin;

			uint hogWindowDim = modelH * modelW * m_params.m_dim;
			size_t length = (m_params.m_propNWords + 1)*m_params.m_numTrWords*modelInstances.second.size();
			Mat features(int(length), hogWindowDim, CV_32F);

			size_t numPos = m_params.m_numTrWords*modelInstances.second.size();
			size_t numNeg = m_params.m_propNWords*numPos;

			std::clog << "Loading " << numPos << " positive examples : " << asciiCode << endl;
			samplePos(features, windowSz, modelInstances.second);
			std::clog << "Loading " << numNeg << " Negative examples : " << asciiCode << endl;
			sampleNeg(asciiCode, features, numPos, negHogFeatures, Size(w, h), Size(modelW, modelH), numNeg);

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
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(ml::SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(features, ml::ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	Mat result;
	svm->predict(features, result);
	double err = 0;
	//TODO: clean this (calcError ?).
	for (size_t i = 0; i < labels.size(); ++i)
	{
		bool labelSign = labels[i] >= 0;
		bool predictedSign = result.at<float>(i) >= 0;
		err += (labelSign != predictedSign);
	}
	clog << "Prediction on training set: " << 1- (err / labels.size()) << endl;
	
	//Ptr<ml::TrainData> trainData = cv::ml::TrainData::create(features, ml::ROW_SAMPLE, Mat(labels));
	//std::clog << "Start training...";
	//svm->setKernel(ml::SVM::LINEAR);
	//svm->trainAuto(trainData);
	//std::clog << "...[done]" << endl;


	string fileName = m_params.m_svmModelsLocation + to_string(asciiCode) + ".yml";
	svm->save(fileName);
}

void LearnModels::getSvmDetector(uchar asciiCode, Mat & sv, double &rho)
{
	Ptr<ml::SVM> svm = m_SVMModels.at(asciiCode);

	// get the support vectors
	sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
}

void LearnModels::drawLocations(Mat &img, const vector<Rect> &locations, const Scalar &color)
{
	for (auto loc : locations)
	{
		rectangle(img, loc, color, 2);
	}
}

void LearnModels::sampleNeg(uchar asciiCode, cv::Mat& features, size_t startPos, const Mat & negHogFeatures, cv::Size imageHogSz, cv::Size modelHogSz, size_t numOfExamples)
{
	int modelH = modelHogSz.height*m_params.m_sbin;
	int modelW = modelHogSz.width*m_params.m_sbin;

	for (const auto& modelInstances : m_modelInstances)
	{
		//TODO: clean this ugliness later.
		if (numOfExamples <= 0)
		{
			break;
		}
		if (asciiCode == modelInstances.first)
		{
			continue;
		}

		for (const auto& mi : modelInstances.second)
		{
			Mat image = m_trainingImages.at(mi.m_pathIm);

			// We expand the query to capture some context
			Rect loc(mi.m_loc.x - m_params.m_sbin, mi.m_loc.y - m_params.m_sbin, mi.m_loc.width + m_params.m_sbin, mi.m_loc.height + m_params.m_sbin);
			
			uint newH = loc.height;
			uint newW = loc.width;

			Mat im = image(loc);

			Mat negPatch;
			resize(im, negPatch, Size((modelHogSz.width + 2)*m_params.m_sbin, (modelHogSz.height + 2)*m_params.m_sbin));

			Mat feat = PedroFeatures::process(negPatch, m_params.m_sbin);
			feat = feat.t();
			feat.convertTo(feat, CV_32F);
			feat.reshape(1, 1).copyTo(features.row(startPos++));

			if (--numOfExamples <= 0)
			{
				break;
			}
		}
	}


	// Pick a random starting cell
	random_device rd;
	mt19937 gen(rd());

	uniform_int_distribution<> byDis(0, imageHogSz.height - modelHogSz.height - 1);
	uniform_int_distribution<> bxDis(0, imageHogSz.width - modelHogSz.width - 1);

	uint hogDim = m_params.m_dim;

	for (; startPos < features.rows; ++startPos)
	{
		int by = byDis(gen);
		int bx = bxDis(gen);
		for (int tmpby = by, i = 0; tmpby < by + modelHogSz.height; ++tmpby, ++i)
		{
			auto sp = tmpby*imageHogSz.width + bx;
			Mat temp = negHogFeatures(Range::all(), Range(sp, sp + modelHogSz.width)).t();
			temp = temp.reshape(1, 1);
			Mat aux = features.colRange(i*modelHogSz.width * hogDim, (i + 1)*modelHogSz.width * hogDim).rowRange(int(startPos), int(startPos + 1));
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

void LearnModels::computeScoresL2(const Mat& HogFeatures, const Mat& w, double rho, int bH, int bW, int dim, int nbinsH, int nbinsW, int step, Mat & score)
{
	for (int by = 0; by <= bH - nbinsH; by += step)
	{
		for (int bx = 0; bx <= bW - nbinsW; bx += step)
		{
			score.at<double>(by,bx) = 0;
			double norm = 0;

			for (int tmpby = by, i = 0; tmpby < by + nbinsH; ++tmpby, ++i)
			{
				int pos = tmpby*bW + bx;
				Mat temp = HogFeatures(Range::all(), Range(pos, pos + nbinsW)).t();
				temp = temp.reshape(1, 1);
				Mat temp2 = w.colRange(i*nbinsW*dim, (i + 1)*nbinsW*dim);
				score.at<double>(by, bx) += temp.dot(temp2);
				norm += temp.dot(temp);
			}
			score.at<double>(by, bx) = (score.at<double>(by, bx)  - rho) / sqrt(norm);
		}
	}
}

cv::Size LearnModels::getHOGWindowSz(uchar asciiCode)
{
	cv::Size sz = m_WindowSz.at(asciiCode);
	return Size(sz.width / m_params.m_sbin, sz.height / m_params.m_sbin);
};
*/

/*
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
}*/
