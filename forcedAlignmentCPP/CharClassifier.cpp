#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>
#include <chrono>
#include <algorithm>
#include "HogUtils.h"
#include "CharClassifier.h"
#include "LibLinearWrapper.h"
#include "TrainingData.h"
#include "JsgdWrapper.h"

using namespace cv;
using namespace std;


CharClassifier::CharClassifier()
	: m_params(Params::getInstance()), 
	m_trData(TrainingData::getInstance()),
	m_docs(m_trData.getTrainingDocs())
{
	m_trData.estimateNormalDistributionParams();
	size_t numDocs = m_docs.size();
	m_numNWords = ceil((double)m_params.m_numNWords / numDocs)*numDocs;
	//computeFeaturesDocs();
}

void CharClassifier::computeFeaturesDocs()
{
	for (size_t i = 0; i < m_docs.size(); ++i)
	{
		clog << "Computing features and labels of doc: " << i << endl;
		m_docs[i].computeFeatures(m_params.m_sbin);
	}
}

void CharClassifier::learnModels()
{
	vector<uchar> asciiCodes = m_trData.getAsciiCodes();
	for (auto asciiCode : asciiCodes)
	{
		HogSvmModel hogSvmModel = learnModel(asciiCode);
		m_svmModels.insert({ asciiCode, move(hogSvmModel)});
	}
}

void CharClassifier::samplePos(const Mat &imDoc, Mat &trHOGs, size_t &position, const Rect &loc, Size sz)
{
	for (auto dx = m_params.m_rangeX.start; dx < m_params.m_rangeX.end; dx += m_params.m_stepSize4PositiveExamples)	{
		for (auto dy = m_params.m_rangeY.start; dy < m_params.m_rangeY.end; dy += m_params.m_stepSize4PositiveExamples)	{
			// Extract image patch
			auto x1 = max(loc.x + dx, 0);
			auto x2 = min(loc.x + loc.width + dx, imDoc.cols - 1);
			auto y1 = max(loc.y + dy, 0);
			auto y2 = min(loc.y + loc.height + dy, imDoc.rows - 1);
			Mat im = imDoc(Rect(x1, y1, x2 - x1, y2 - y1));

			Mat posPatch;
			if (im.rows != sz.height || im.cols != sz.width)
			{
				resize(im, posPatch, sz);
			}
			else
			{
				im.copyTo(posPatch);
			}

			Mat feat = HogUtils::process(posPatch, m_params.m_sbin);
			feat.convertTo(feat, CV_32F);
			feat.reshape(1, 1).copyTo(trHOGs.row(position++));
		}
	}
}

void CharClassifier::sampleNeg(Mat &trHOGs, size_t position, int wordsByDoc, const HogSvmModel &hs_model)
{
	random_device rd;
	mt19937 gen(rd());

	uint dim = m_params.m_dim;
	size_t stepSize = hs_model.m_bW*dim;
	uint sbin = m_params.m_sbin;

	for (uint id = 0; id < m_docs.size(); ++id)
	{
		Mat fD;
		int BH, BW;
		m_docs[id].getComputedFeatures(fD, BH, BW, sbin);

		float *flat = fD.ptr<float>(0);

		for (int jj = 0; jj < wordsByDoc; ++jj)
		{
			// Pick a random starting cell
			uniform_int_distribution<> byDis(0, BH - hs_model.m_bH - 1);
			uniform_int_distribution<> bxDis(0, BW - hs_model.m_bW - 1);

			int by = byDis(gen);
			int bx = bxDis(gen);
			auto *flat_trHOGs = trHOGs.ptr<float>(position);

			for (auto tmpby = by, i = 0; tmpby < by + hs_model.m_bH; ++tmpby, ++i)
			{
				auto pos = (tmpby*BW + bx)*dim;
				copy(flat + pos, flat + pos + stepSize, flat_trHOGs + i*stepSize);
			}
			++position;
		}
	}
}

void CharClassifier::trainClassifier(const Mat &trHOGs, HogSvmModel &hs_model, size_t numSamples, size_t numTrWords, string svmlib)
{
	random_device rd;
	mt19937 gen(rd());

	Mat labels = Mat::ones(numSamples, 1, CV_32F);
	labels.rowRange(0, numTrWords) = 0;

	if (m_params.m_svmlib == "jsgd")
	{
#if _WIN32 || _WIN64
		cerr << "JSGD option is not supported on windows OS" << endl;
#elif __GNUC__
		std::vector<int> randp;
		randp.reserve(numSamples);
		int n(0);
		std::generate_n(std::back_inserter(randp), numSamples, [n]()mutable { return n++; });
		std::shuffle(randp.begin(), randp.end(), gen);
		Mat trHOGs_shuffled(trHOGs.size(), CV_32F);
		Mat labels_shuffled(labels.size(), CV_32S);

		for (size_t i = 0; i < randp.size(); ++i)
		{
			trHOGs.row(randp[i]).copyTo(trHOGs_shuffled.row(i));
			labels_shuffled.at<int>(i) = static_cast<int>(labels.at<float>(randp[i]));
		}
		JsgdWrapper::trainModel(labels_shuffled, trHOGs_shuffled, hs_model.weight);
#endif
	}
	else if (m_params.m_svmlib == "liblinear")
	{
		LibLinearWrapper ll;
		ll.trainModel(labels, trHOGs, hs_model.weight);
	}
	else if (m_params.m_svmlib == "bl")
	{
		auto *ptr = trHOGs.ptr<float>(m_params.m_numTrWords / 2);
		hs_model.weight.assign(ptr, ptr + trHOGs.cols);
	}
}

void CharClassifier::NormalizeFeatures(cv::Mat & features)
{
	Mat temp, rp;
	for (int i = 0; i < features.rows; ++i)
	{
		double rowNorm = norm(features.row(i));
		if (rowNorm > 0)
		{
			features.row(i) = features.row(i) * (1 / rowNorm);
		}
	}
}

HogSvmModel CharClassifier::learnModel(uchar asciiCode)
{
	HogSvmModel hs_model(asciiCode, m_params.m_pathCharModels);

	auto iter = m_svmModels.find(asciiCode);
	if (iter != m_svmModels.end())
	{
		hs_model = iter->second;
		return hs_model;
	}
	else if (hs_model.loadFromFile())
	{
		return hs_model;
	}

	clog << "computing model for " << asciiCode << endl;

	uint pxbin = m_params.m_sbin;
	
	const auto& chVec = m_trData.getSamples(asciiCode);

	int modelNew_H = round(m_trData.getMeanHeight(asciiCode)) + 2 * pxbin;
	int modelNew_W = round(m_trData.getMeanWidth(asciiCode)) + 2 * pxbin;

	uint res;
	while ((res = (modelNew_H % pxbin)))
	{
		modelNew_H -= res;
	}
	while ((res = (modelNew_W % pxbin)))
	{
		modelNew_W -= res;
	}

	hs_model.m_newH = modelNew_H;
	hs_model.m_newW = modelNew_W;
	hs_model.m_bH = modelNew_H / pxbin - 2;
	hs_model.m_bW = modelNew_W / pxbin - 2;
	uint descsz = hs_model.m_bH*hs_model.m_bW*m_params.m_dim;

	int numSamples = (m_params.m_numTrWords + m_numNWords)*chVec.size();
	clog << "num of training samples for " << asciiCode << ": " << numSamples << endl;
	if (numSamples != 0)
	{
	Mat trHOGs = Mat::zeros(numSamples, descsz, CV_32F);
	
	size_t position = 0;
	
	for (const auto& ci : chVec)
	{
		// We expand the query to capture some context
		Rect loc(ci.m_loc.x - pxbin, ci.m_loc.y - pxbin, ci.m_loc.width + 2 * pxbin, ci.m_loc.height + 2 * pxbin);

		Mat imDoc = m_docs[ci.m_docNum].m_origImage;

		// Get positive windows
		samplePos(imDoc, trHOGs, position, loc, Size(modelNew_W, modelNew_H));
	}

	// Get negative windows
	int wordsByDoc = (m_numNWords*chVec.size()) / m_docs.size();
	sampleNeg(trHOGs, position, wordsByDoc, hs_model);

	// Apply L2 - norm.
	NormalizeFeatures(trHOGs);
	
	string svmlib = (chVec.size() == 1) ? "bl" : m_params.m_svmlib;
	trainClassifier(trHOGs, hs_model, numSamples, m_params.m_numTrWords*chVec.size(), svmlib);

		hs_model.setInitialized();
	hs_model.save2File();
	}
	return hs_model;
}

HogSvmModel CharClassifier::learnExemplarModel(const Doc& doc, const Character& query)
{
	HogSvmModel hs_model(query.m_asciiCode, m_params.m_pathCharModels);

	Rect loc = Character::resizeChar(query.m_loc, m_params.m_sbin);

	hs_model.m_newH = loc.height;
	hs_model.m_newW = loc.width;
	hs_model.m_bH = hs_model.m_newH / m_params.m_sbin - 2;
	hs_model.m_bW = hs_model.m_newW / m_params.m_sbin - 2;
	uint descsz = hs_model.m_bH*hs_model.m_bW*m_params.m_dim;

	Mat trHOGs = Mat::zeros(m_params.m_numTrWords + m_numNWords, descsz, CV_32F);

	Mat imDoc = doc.m_origImage;

	// Get positive windows
	size_t ps = 0;
	samplePos(imDoc, trHOGs, ps, loc, Size(hs_model.m_newW, hs_model.m_newH));

	// Get negative windows
	int wordsByDoc = m_numNWords / m_docs.size();
	uint startPos = m_params.m_numTrWords;
	sampleNeg(trHOGs, startPos, wordsByDoc, hs_model);

	// Apply L2 - norm.
	NormalizeFeatures(trHOGs);

	// Train the (SVM) Classifier
	int numSamples = m_params.m_numTrWords + m_numNWords;
	trainClassifier(trHOGs, hs_model, numSamples, m_params.m_numTrWords, m_params.m_svmlib);
	
	return hs_model;
}
