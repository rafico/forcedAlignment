#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include <numeric>
#include "CharSpotting.h"
#include "HogUtils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace boost::filesystem;

CharSpotting::CharSpotting()
	: m_numRelevantWordsByClass(UCHAR_MAX + 1, 0), 
	m_trainingDocs(m_classifier.getTrainingDocs()),
	m_params(Params::getInstance())
{
	loadData();
}

void CharSpotting::loadData()
{
	uint classNum = 0;
	for (size_t docNum = 0; docNum < m_trainingDocs.size(); ++docNum)
	{
		for (auto &ch : m_trainingDocs[docNum].m_chars)
		{
			auto iter = m_classes.find(ch.m_asciiCode);
			if (iter != m_classes.end())
			{
				classNum = iter->second;
			}
			else
			{
				classNum = uint(m_classes.size());
				m_classes.insert({ ch.m_asciiCode, classNum });
			}
			++m_numRelevantWordsByClass[classNum];
		}
	}

	auto iter = find_if(m_numRelevantWordsByClass.begin(), m_numRelevantWordsByClass.end(), [](uint cnt){return cnt == 0; });
	m_numClasses = distance(m_numRelevantWordsByClass.begin(), iter);
	m_numRelevantWordsByClass.resize(m_numClasses);

	size_t numDocs = m_trainingDocs.size();
	m_relevantBoxesByClass.resize(numDocs*m_numClasses);
	for (size_t i = 0; i < numDocs; ++i)
	{
		const auto& chars = m_trainingDocs[i].m_chars;
		for (auto &ch : chars)
		{
			uint classNum = m_classes[ch.m_asciiCode];
			Rect loc = ch.m_loc;
			size_t idx = i*m_numClasses + classNum;
			m_relevantBoxesByClass[idx].push_back(loc);
		}
	}
}

void CharSpotting::evaluateModels(bool ExemplarModel /*= false */)
{
	size_t index = 1;
	char buffer[256];

	for (const auto& doc : m_trainingDocs)
	{
		for (const auto& query : doc.m_chars)
		{
			uint classNum = m_classes[query.m_asciiCode];
			uint nrelW = m_numRelevantWordsByClass[classNum];

			HogSvmModel hogSvmModel = ExemplarModel ? m_classifier.learnExemplarModel(doc, query) : m_classifier.learnModel(query.m_asciiCode);

			vector<double> scores;
			vector<double> resultLabels;
			vector<pair<Rect, size_t>> locWords;
			evalModel(hogSvmModel, classNum, scores, resultLabels, locWords);
			saveResultImages(doc, query, resultLabels, locWords);
			double mAP, rec;
			compute_mAP(resultLabels, nrelW, mAP, rec);
			sprintf(buffer, "%20c (%5lu): %2.2f (mAP) nRel: %d", query.m_asciiCode, index, mAP, nrelW);
			std::clog << buffer << endl;
			++index;
		}
	}
}

void CharSpotting::evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords)
{
	vector<double> scoresWindows;
	vector<bool> resultsWindows;
	vector<pair<Rect, size_t>> locWindows;

	for (size_t docNum = 0; docNum < m_trainingDocs.size(); ++docNum)
	{
		const Doc& doc = m_trainingDocs[docNum];
		vector<Rect> locW;
		vector<double> scsW;
		HogUtils::getWindows(doc, hs_model, scsW, locW, m_params.m_step, m_params.m_sbin);

		for_each(scsW.begin(), scsW.end(), [](double &scs){if (std::isnan(scs)) scs = -1; });

		Mat I;
		sortIdx(Mat(scsW), I, cv::SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
		I = I.rowRange(I.rows - m_params.m_thrWindows - 1, I.rows);
		vector<int> pick = HogUtils::nms(I, locW, m_params.m_overlapnms);

		Mat scsW_(1, int(pick.size()), CV_64F);
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
		for_each(locW_.begin(), locW_.end(), [&](const Rect& locW){locWindows.push_back(std::make_pair(locW, docNum)); });
	}

	Mat scoresIdx;
	sortIdx(Mat(scoresWindows), scoresIdx, cv::SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
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

void CharSpotting::saveResultImages(const Doc& doc, const Character& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords)
{
	string qPathString = m_params.m_pathResultsImages + std::to_string(query.m_globalIdx + 1) + "/";
	path p(qPathString);
	if (!exists(p))
	{
		boost::filesystem::create_directory(p);
	}
	Mat imDocQ = doc.m_origImage;
	Mat imq = imDocQ(query.m_loc);
	imwrite(qPathString + "000q.png", imq);
	size_t numIm = std::min(resultLabels.size(), m_params.m_numResultImages);
	char fileName[128];
	for (size_t i = 0; i < numIm; ++i)
	{
		char flag = resultLabels[i] ? 'c' : 'e';
		sprintf(fileName, "%.3lu%c.png", i + 1, flag);
		Rect bb = locWords[i].first;
		size_t docIdx = locWords[i].second;
		Mat imDoc = m_trainingDocs[docIdx].m_origImage;

		auto x1 = std::max(bb.x, 0);
		auto x2 = std::min(bb.x + bb.width, imDoc.cols - 1);
		auto y1 = std::max(bb.y, 0);
		auto y2 = std::min(bb.y + bb.height, imDoc.rows - 1);
		Mat im = imDoc(Rect(x1, y1, x2 - x1, y2 - y1));

		imwrite(qPathString + fileName, im);
	}
}

void CharSpotting::compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec)
{
	// Compute the mAP
	vector<double> precAt(resultLabels.size());
	double sum = 0;
	for (size_t i = 0; i < resultLabels.size(); ++i)
	{
		sum += resultLabels[i];
		precAt[i] = sum / (i + 1);
	}
	mAP = std::inner_product(precAt.begin(), precAt.end(), resultLabels.begin(), .0);
	mAP /= nrelW;
	rec = sum / nrelW;
}