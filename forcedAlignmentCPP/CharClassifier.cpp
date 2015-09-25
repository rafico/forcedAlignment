#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include <numeric>
#include <chrono>
#include <algorithm>
#include "HogUtils.h"
#include "CharClassifier.h"
#include "LibLinearWrapper.h"
#include "TrainingData.h"
//#include "JsgdWrapper.h"


using namespace boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

CharClassifier::CharClassifier()
	: m_params(Params::getInstance()), m_numRelevantWordsByClass(UCHAR_MAX + 1, 0), m_trData(m_params)
{
	loadTrainingData();
	m_trData.computeNormalDistributionParams();
	computeFeaturesDocs();
}

void CharClassifier::loadTrainingData()
{
	m_docs = m_trData.m_trainingDocs;
	m_numRelevantWordsByClass = m_trData.m_numRelevantWordsByClass;
	m_relevantBoxesByClass = m_trData.m_relevantBoxesByClass;
	m_numClasses = m_trData.m_numRelevantWordsByClass.size();
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
	const auto& t_chs = m_trData.m_charInstances;
	for (auto& ch : t_chs)
	{
		uchar asciiCode = ch.first;
		HogSvmModel hogSvmModel = learnModel(asciiCode);
		m_svmModels.insert({ asciiCode, move(hogSvmModel)});
	}
}

void CharClassifier::evaluateModels(bool ExemplarModel/*  = true */)
{
	size_t index = 1;
	char buffer[256];

	for (const auto& doc : m_docs)
	{
		for (const auto& query : doc.m_chars)
		{
			uint classNum = m_trData.getCharClass(query.m_asciiCode);
			uint nrelW = m_numRelevantWordsByClass[classNum];
			
			HogSvmModel hogSvmModel = ExemplarModel ? learnExemplarModel(doc, query) : learnModel(query.m_asciiCode);
			
			vector<double> scores;
			vector<double> resultLabels;
			vector<pair<Rect, size_t>> locWords;
			evalModel(hogSvmModel, classNum, scores, resultLabels, locWords);
			saveResultImages(doc, query, resultLabels, locWords);
			double mAP, rec;
			compute_mAP(resultLabels, nrelW, mAP, rec);
			sprintf(buffer, "%20c (%5lu): %2.2f (mAP) nRel: %d", query.m_asciiCode, index, mAP, nrelW);
			clog << buffer << endl;
			++index;
		}
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

	for (uint id = 0; id < m_docs.size(); ++id)
	{
		Mat fD = m_docs[id].m_features;
		uint BH = m_docs[id].m_bH;
		uint BW = m_docs[id].m_bW;

		float *flat = fD.ptr<float>(0);

		for (int jj = 0; jj < wordsByDoc; ++jj)
		{
			// Pick a random starting cell
			uniform_int_distribution<> byDis(0, BH - hs_model.m_bH - 1);
			uniform_int_distribution<> bxDis(0, BW - hs_model.m_bW - 1);

			int by = byDis(gen);
			int bx = bxDis(gen);
			auto *flat_trHOGs = trHOGs.ptr<float>(position);

			for (size_t tmpby = by, i = 0; tmpby < by + hs_model.m_bH; ++tmpby, ++i)
			{
				size_t pos = (tmpby*BW + bx)*dim;
				copy(flat + pos, flat + pos + stepSize, flat_trHOGs + i*stepSize);
			}
			++position;
		}
	}
}

void CharClassifier::trainClassifier(const Mat &trHOGs, HogSvmModel &hs_model, size_t numSamples, size_t numTrWords)
{
	random_device rd;
	mt19937 gen(rd());

	Mat labels = Mat::ones(numSamples, 1, CV_32F);
	labels.rowRange(0, numTrWords) = 0;

	if (m_params.m_svmlib == "jsgd")
	{
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
		//JsgdWrapper::trainModel(labels_shuffled, trHOGs_shuffled, hs_model.weight);
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

	int numSamples = (m_params.m_numTrWords + m_params.m_numNWords)*chVec.size();
	clog << "num of training samples for " << asciiCode << ": " << numSamples << endl;
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
	int wordsByDoc = (m_params.m_numNWords*chVec.size()) / m_docs.size();
	sampleNeg(trHOGs, position, wordsByDoc, hs_model);

	// Apply L2 - norm.
	NormalizeFeatures(trHOGs);
	
	trainClassifier(trHOGs, hs_model, numSamples, m_params.m_numTrWords*chVec.size());

	hs_model.save2File();
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

	Mat trHOGs = Mat::zeros(m_params.m_numTrWords + m_params.m_numNWords, descsz, CV_32F);

	Mat imDoc = doc.m_origImage;

	// Get positive windows
	size_t ps = 0;
	samplePos(imDoc, trHOGs, ps, loc, Size(hs_model.m_newW, hs_model.m_newH));

	// Get negative windows
	int wordsByDoc = m_params.m_numNWords / m_docs.size();
	uint startPos = m_params.m_numTrWords;

	// Apply L2 - norm.
	NormalizeFeatures(trHOGs);

	// Train the (SVM) Classifier
	int numSamples = m_params.m_numTrWords + m_params.m_numNWords;
	trainClassifier(trHOGs, hs_model, numSamples, m_params.m_numTrWords);
	
	return hs_model;
}

void CharClassifier::evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords)
{
	vector<double> scoresWindows;
	vector<bool> resultsWindows;
	vector<pair<Rect, size_t>> locWindows;

	for (size_t docNum = 0; docNum < m_docs.size(); ++docNum)
	{
		const Doc& doc = m_docs[docNum];
		vector<Rect> locW;
		vector<double> scsW;
		HogUtils::getWindows(doc, hs_model, scsW, locW, m_params.m_step, m_params.m_sbin);
		
		for_each(scsW.begin(), scsW.end(), [](double &scs){if (std::isnan(scs)) scs = -1; });

		Mat I;
		sortIdx(Mat(scsW), I, SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
		I = I.rowRange(I.rows - m_params.m_thrWindows-1, I.rows);
		vector<int> pick = HogUtils::nms(I, locW, m_params.m_overlapnms);

		Mat scsW_(1, int(pick.size()),CV_64F);
		vector<Rect> locW_(pick.size());
		for (uint i = 0; i < pick.size(); ++i)
		{
			scsW_.at<double>(i) = scsW[pick[i]];
			locW_[i] = locW[pick[i]];
		}
		const vector<Rect>& relBoxes = getRelevantBoxesByClass(classNum, docNum);

		vector<bool> res(pick.size(), false);
		if (!(relBoxes.empty()))
		{
			auto areaP = hs_model.m_newH*hs_model.m_newW;
			for (uint i = 0; i < locW_.size(); ++i)
			{
				for (uint j = 0; j < relBoxes.size(); ++j)
				{
					auto areaGT = relBoxes[j].area();
					double intArea = (locW_[i] & relBoxes[j]).area();
					double denom = (areaGT + areaP) - intArea;
					double overlap = intArea / denom;
					if (overlap >= m_params.m_overlap)
					{
						res[i] = true;
						continue;
					}
				}
			}
		}
		double *ptr_scsW = scsW_.ptr<double>(0);
		scoresWindows.insert(scoresWindows.end(), ptr_scsW, ptr_scsW + scsW_.cols);
		resultsWindows.insert(resultsWindows.end(), res.begin(), res.end());
		for_each(locW_.begin(), locW_.end(), [&](const Rect& locW){locWindows.push_back(make_pair(locW,docNum)); });
	}

	Mat scoresIdx;
	sortIdx(Mat(scoresWindows), scoresIdx, SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
	scores.resize(scoresIdx.rows);
	resultLabels.resize(scoresIdx.rows);
	locWords.resize(scoresIdx.rows);
	size_t lastPosElement = 0;
	for (int i = 0; i < scoresIdx.rows; ++i)
	{
		int idx = scoresIdx.at<int>(i);
		scores[i] = scoresWindows[idx];
		locWords[i] = locWindows[idx];
		if ((resultLabels[i] = resultsWindows[idx]))
		{
			lastPosElement = i;
		}
	}
	// Suppress the last non - relevant windows(does not affect mAP)
	scores.resize(lastPosElement + 1);
	resultLabels.resize(lastPosElement + 1);
	locWords.resize(lastPosElement + 1);
}

void CharClassifier::saveResultImages(const Doc& doc, const Character& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords)
{
	string qPathString = m_params.m_pathResultsImages + to_string(query.m_globalIdx+1) + "/";
	path p(qPathString);
	if (!exists(p))
	{
		boost::filesystem::create_directory(p);
	}
	Mat imDocQ = doc.m_origImage;
	Mat imq = imDocQ(query.m_loc);
	imwrite(qPathString + "000q.png", imq);
	size_t numIm = min(resultLabels.size(), m_params.m_numResultImages);
	char fileName[128];
	for (size_t i = 0; i < numIm; ++i)
	{
		char flag = resultLabels[i] ? 'c' : 'e';
		sprintf(fileName, "%.3lu%c.png", i + 1, flag);
		Rect bb = locWords[i].first;
		size_t docIdx = locWords[i].second;
		Mat imDoc = m_docs[docIdx].m_origImage;
		
		auto x1 = max(bb.x, 0);
		auto x2 = min(bb.x + bb.width, imDoc.cols - 1);
		auto y1 = max(bb.y, 0);
		auto y2 = min(bb.y + bb.height, imDoc.rows - 1);
		Mat im = imDoc(Rect(x1, y1, x2 - x1, y2 - y1));

		imwrite(qPathString + fileName, im);
	}
}

void CharClassifier::compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec)
{
	// Compute the mAP
	vector<double> precAt(resultLabels.size());
	double sum = 0;
	for (size_t i = 0; i < resultLabels.size(); ++i)
	{
		sum += resultLabels[i];
		precAt[i] = sum / (i + 1);
	}
	mAP = inner_product(precAt.begin(), precAt.end(), resultLabels.begin(), .0);
	mAP /= nrelW;
	rec = sum / nrelW;
}

void CharClassifier::load_char_stats(charStatType &meanCont, charStatType& stdCont) const
{
	m_trData.load_char_stats(meanCont, stdCont);
}

void CharClassifier::getMinMaxCharLength(uint &maxCharLen, uint &minCharLen)
{
	maxCharLen = m_trData.m_globalMaxWidth;
	minCharLen = m_trData.m_globalMinWidth;
}
