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
#include "PedroFeatures.h"
#include "LearnModels.h"
#include "LibLinearWrapper.h"
#include "JsgdWrapper.h"

using namespace boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

LearnModels::LearnModels()
	: m_numRelevantWordsByClass(UCHAR_MAX+1, 0)
{
	m_params.initDirs();
	loadTrainingData();
	computeFeaturesDocs();
}

void LearnModels::loadTrainingData()
{
	clog << "* Loading queries *" << endl;
	clog << "* Computing queries *" << endl;
	clog << "* Loading documents *" << endl;
	clog << "* Initializing test documents *" << endl;

	path dir_path(m_params.m_pathDocuments);
	if (!exists(dir_path))
	{
		cerr << "Error: Unable to read models from " << m_params.m_pathDocuments << std::endl;
		return;
	}

	uint globalIdx=0;

	vector<path> vp; // store paths, so we can sort them later
	copy(directory_iterator(dir_path), directory_iterator(), back_inserter(vp));
	sort(vp.begin(), vp.end()); // sort, since directory iteration is not ordered on some file systems

	for (const auto &itr : vp)
	{
		if (itr.extension() == ".csv")
		{
			cout << "Loading " << itr.filename() << endl;
			std::ifstream str(itr.string());
			if (!str.good())
			{
				std::cerr << "Error: Unable to read models from " << itr.string() << std::endl;
			}
			
			string fileName = itr.filename().stem().string();
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
	m_numClasses = distance(m_numRelevantWordsByClass.begin(), iter);
	m_numRelevantWordsByClass.resize(m_numClasses);
	
	m_params.m_numDocs = m_docs.size();
	m_relevantBoxesByClass.resize(m_params.m_numDocs*m_numClasses);
	for (size_t i = 0; i < m_params.m_numDocs; ++i)
	{
		const auto& chars = m_docs[i].m_chars;
		for (const auto& ch : chars)
		{
			uint classNum = ch.m_classNum;
			Rect loc = ch.m_loc;
			size_t idx = i*m_numClasses + classNum;
			m_relevantBoxesByClass[idx].push_back(loc);
		}
	}
	m_params.m_numNWords = ceil((double)m_params.m_numNWords / m_params.m_numDocs)*m_params.m_numDocs;
}

void LearnModels::getImagesDocs()
{
	clog << "* Getting images and resizing *" << endl;
	for (auto& doc : m_docs)
	{
		uint H = doc.m_H;
		uint W = doc.m_W;
		uint res;

		while ((res = (H % m_params.m_sbin)))
		{
			H -= res;
		}
		while ((res = (W % m_params.m_sbin)))
		{
			W -= res;
		}
		uint difH = doc.m_H - H;
		uint difW = doc.m_W - W;
		uint padYini = difH / 2;
		uint padYend = difH - padYini;
		uint padXini = difW / 2;
		uint padXend = difW - padXini;
		
		const Mat &im = doc.m_origImage;
		im(Range(padYini, im.rows - padYend), Range(padXini, im.cols - padXend)).copyTo(doc.m_image);
		doc.m_yIni = padYini;
		doc.m_xIni = padXini;
		doc.m_H = doc.m_image.rows;
		doc.m_W = doc.m_image.cols;
	}
}


void LearnModels::computeFeaturesDocs()
{
	getImagesDocs();
	for (size_t i = 0; i < m_docs.size(); ++i)
	{
		clog << "Computing features and labels of doc " << i << endl;
		int bH, bW;
		Mat feat = PedroFeatures::process(m_docs[i].m_image, m_params.m_sbin, &bH, &bW);
		feat.convertTo(m_docs[i].m_features, CV_32F);
		m_docs[i].m_bH = bH;
		m_docs[i].m_bW = bW;
	}
}

void LearnModels::LearnModelsAndEvaluate()
{
	size_t index = 1;
	char buffer[256];

	for (const auto& doc : m_docs)
	{
		for (const auto& query : doc.m_chars)
		{
			uint classNum = query.m_classNum;
			uint nrelW = m_numRelevantWordsByClass[classNum];
			HogSvmModel hogSvmModel;
			learnModel(doc, query, hogSvmModel);
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

void LearnModels::learnModel(const Doc& doc, const CharInstance& ci, HogSvmModel &hs_model)
{
	uint pxbin = m_params.m_sbin;

	// We expand the query to capture some context
	Rect loc(ci.m_loc.x - pxbin, ci.m_loc.y - pxbin, ci.m_loc.width + 2*pxbin, ci.m_loc.height + 2*pxbin);
	int modelNew_H = loc.height;
	int modelNew_W = loc.width;

	uint res;
	while ((res = (modelNew_H % pxbin)))
	{
		modelNew_H -= res;
		loc.height -= res;
		loc.y += int(floor(double(res) / 2));
	}
	while ((res = (modelNew_W % pxbin)))
	{
		modelNew_W -= res;
		loc.width -= res;
		loc.x += int(floor(double(res) / 2));
	}

	hs_model.m_newH = modelNew_H;
	hs_model.m_newW = modelNew_W;
	hs_model.m_bH = modelNew_H / m_params.m_sbin - 2;
	hs_model.m_bW = modelNew_W / m_params.m_sbin - 2;
	uint descsz = hs_model.m_bH*hs_model.m_bW*m_params.m_dim;

	Mat trHOGs = Mat::zeros(m_params.m_numTrWords + m_params.m_numNWords, descsz, CV_64F);
	
	Mat imDoc = doc.m_origImage;
	
	// Get positive windows
	uint ps = 0;
	for (auto dx = m_params.m_rangeX.start; dx < m_params.m_rangeX.end; dx += m_params.m_stepSize4PositiveExamples)	{
		for (auto dy = m_params.m_rangeY.start; dy < m_params.m_rangeY.end; dy += m_params.m_stepSize4PositiveExamples)	{
			// Extract image patch
			auto x1 = max(loc.x + dx, 0);
			auto x2 = min(loc.x + loc.width + dx, imDoc.cols - 1);
			auto y1 = max(loc.y + dy, 0);
			auto y2 = min(loc.y + loc.height + dy, imDoc.rows - 1);
			Mat im = imDoc(Rect(x1, y1, x2 - x1, y2 - y1));

			Mat posPatch;
			if (im.rows != modelNew_H || im.cols != modelNew_W)
			{
				resize(im, posPatch, Size(modelNew_W, modelNew_H));
			}
			else
			{
				im.copyTo(posPatch);
			}

			Mat feat = PedroFeatures::process(posPatch, m_params.m_sbin);
			feat.reshape(1, 1).copyTo(trHOGs.row(ps++));
		}
	}
	// Get negative windows
	random_device rd;
	mt19937 gen(rd());

	int wordsByDoc = m_params.m_numNWords / m_params.m_numDocs;
	uint startPos = m_params.m_numTrWords;
	
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
			auto *flat_trHOGs = trHOGs.ptr<double>(startPos);

			for (size_t tmpby = by, i = 0; tmpby < by + hs_model.m_bH; ++tmpby, ++i)
			{
				size_t pos = (tmpby*BW + bx)*dim;
				copy(flat + pos, flat + pos + stepSize, flat_trHOGs + i*stepSize);
			}
			++startPos;
		}
	}

	// Apply L2 - norm.
	NormalizeFeatures(trHOGs);
	trHOGs.convertTo(trHOGs, CV_32F);
		
	int numSamples = m_params.m_numTrWords + m_params.m_numNWords;
	Mat labels = Mat::ones(numSamples, 1, CV_32F);
	labels.rowRange(0,m_params.m_numTrWords) = 0;
	
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
		JsgdWrapper::trainModel(labels_shuffled, trHOGs_shuffled, hs_model.weight);
	}
	else if (m_params.m_svmlib == "liblinear")
	{
		LibLinearWrapper ll;
		ll.trainModel(labels, trHOGs, hs_model.weight);
	}
	else if (m_params.m_svmlib == "bl")
	{
		auto *ptr = trHOGs.ptr<float>(m_params.m_numTrWords / 2 - 1);
		hs_model.weight.assign(ptr, ptr + trHOGs.cols);
	}
}

void LearnModels::NormalizeFeatures(cv::Mat & features)
{
	Mat temp, rp;
	for (int i = 0; i < features.rows; ++i)
	{
		double rowNorm = norm(features.row(i));
		if (rowNorm > 0)
		{
			features.row(i) = features.row(i) * (1/rowNorm);
		}
	}
}

void LearnModels::evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords)
{
	vector<double> scoresWindows;
	vector<bool> resultsWindows;
	vector<pair<Rect, size_t>> locWindows;

	for (size_t docNum = 0; docNum < m_docs.size(); ++docNum)
	{
		const Doc& doc = m_docs[docNum];
		vector<Rect> locW;
		vector<double> scsW;
		getWindows(doc, hs_model, scsW, locW);
		
		for_each(scsW.begin(), scsW.end(), [](double &scs){if (std::isnan(scs)) scs = -1; });

		Mat I;
		sortIdx(Mat(scsW), I, SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
		I = I.rowRange(I.rows - m_params.m_thrWindows-1, I.rows);
		vector<int> pick = nms(I, locW, m_params.m_overlapnms);

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
	// Supress the last non - relevant windows(does not affect mAP)
	scores.resize(lastPosElement + 1);
	resultLabels.resize(lastPosElement + 1);
	locWords.resize(lastPosElement + 1);
}

void LearnModels::getWindows(const Doc& doc, const HogSvmModel& hs_model, vector<double>& scsW, vector<Rect>& locW)
{
	//TODO: optimize this function (try using GPU and/or flattening the Mat).

	// Extracts all the possible patches from a document images and computes the
	// score given the queried word model

	Mat featDoc = doc.m_features;
	CV_Assert(featDoc.isContinuous());
	float *flat = featDoc.ptr<float>(0);
	auto bH = doc.m_bH;
	auto bW = doc.m_bW;
	auto nbinsH = hs_model.m_bH; 
	auto nbinsW = hs_model.m_bW;
	auto dim = hs_model.weight.size() / (nbinsH*nbinsW);
	auto numWindows = (bH - nbinsH + 1) * (bW - nbinsW + 1);
	uint step = m_params.m_step;
	scsW.resize(numWindows);

	size_t k = 0;
	for (size_t by = 0; by <= bH - nbinsH; by += step)
	{
		for (size_t bx = 0; bx <= bW - nbinsW; bx += step)
		{
			scsW[k] = 0;
			double norm = 0;

			for (size_t tmpby = by, i = 0; tmpby < by + nbinsH; ++tmpby, ++i)
			{
				size_t pos = (tmpby*bW + bx)*dim;
				scsW[k] += inner_product(hs_model.weight.data() + i*nbinsW*dim, hs_model.weight.data() + (i + 1)*nbinsW*dim, flat + pos, .0);
				norm += inner_product(flat + pos, flat + pos + nbinsW*dim, flat + pos, .0);
			}
			scsW[k++] /= sqrt(norm);
			locW.push_back(Rect(bx*m_params.m_sbin + doc.m_xIni, by*m_params.m_sbin + doc.m_yIni, (nbinsW+2)*m_params.m_sbin, (nbinsH+2)*m_params.m_sbin));
		}
	}
}

vector<int> LearnModels::nms(Mat I, const vector<Rect>& X, double overlap)
{
	int i, j;
	int N = I.rows;

	vector<int> used(N,0);
	vector<int> pick(N,-1);

	int Npick = 0;
	for (i = N - 1; i >= 0; i--)
	{
		if (!used[i])
		{
			used[i] = 1;
			pick[Npick++] = I.at<int>(i);
			for (j = 0; j < i; ++j)
			{
				const Rect& r_i = X[I.at<int>(i)];
				const Rect& r_j = X[I.at<int>(j)];
				if (!used[j] && (double((r_i & r_j).area()) / r_j.area()) >= overlap)
				{
					used[j] = 1;
				}
			}
		}
	}
	pick.resize(Npick);
	return pick;
}

void LearnModels::saveResultImages(const Doc& doc, const CharInstance& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords)
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
		const Rect& bb = locWords[i].first;
		size_t docIdx = locWords[i].second;
		Mat im = m_docs[docIdx].m_image(bb);
		imwrite(qPathString + fileName, im);
	}
}

void LearnModels::compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec)
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
