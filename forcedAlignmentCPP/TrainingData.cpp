#include <numeric>
#include "TrainingData.h"
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace boost::filesystem;
using namespace std;

TrainingData::TrainingData()
	: m_params(Params::getInstance())
{
	std::clog << "* Loading documents *" << endl;

	path dir_path(m_params.m_pathGT);
	if (!exists(dir_path))
	{
		cerr << "Error: Directory " << m_params.m_pathGT << " does not exist" << std::endl;
		return;
	}

	StringVector training_file_list;
	StringVector validation_file_list;
	training_file_list.read(m_params.m_pathTrainingFiles);
	validation_file_list.read(m_params.m_pathValidationFiles);

	loadDocs(training_file_list, m_trainingDocs, m_file2trDoc);
	loadDocs(validation_file_list, m_trainingDocs, m_file2trDoc);

	combineChars();
	//displayTrainingData();
	//writeQueriesAndDocsGTPfiles();
}

void TrainingData::loadDocs(const StringVector& file_list, vector<Doc> &docCont, unordered_map<string, size_t> &file2DocMap)
{
	for (const auto &fileName : file_list)
	{
		cout << "Loading " << fileName << endl;
		string name = path(fileName).stem().string();
		string imgFileName = m_params.m_pathImages + name + ".jpg";
		Doc doc(imgFileName);
		doc.loadXml(m_params.m_pathGT + fileName);
		docCont.push_back(Doc(doc));
		file2DocMap.insert({ name, docCont.size() - 1 });
	}
}

void TrainingData::combineChars()
{
	for (size_t docNum = 0; docNum < m_trainingDocs.size(); ++docNum)
	{
		for (auto &ch : m_trainingDocs[docNum].m_chars)
		{
			ch.m_docNum = docNum;
			auto iter = m_charInstances.find(ch.m_asciiCode);
			if (iter != m_charInstances.end())
			{
				iter->second.m_instances.push_back(ch);
			}
			else
			{
				TrainingCharsHelper tch;
				tch.m_instances.push_back(ch);
				m_charInstances.insert({ ch.m_asciiCode, tch });
			}
		}
	}
}

vector<uchar> TrainingData::getAsciiCodes()
{
	vector<uchar> v;
	for_each(m_charInstances.begin(), m_charInstances.end(), [&](TrainingCharsCont::value_type &csType){v.push_back(csType.first); });
	return v;
}

const vector<Character>& TrainingData::getSamples(uint asciiCode)
{
	auto& tch = m_charInstances[asciiCode];
	return tch.m_instances;
}

void TrainingData::getExtermalWidths(vector<uchar>& charSeq, int &maxWidth, int& minWidth)
{
	maxWidth = 0;
	minWidth = 5000;

	for (auto asciiCode : charSeq)
	{
		if (asciiCode != '|')
		{
		auto& tch = m_charInstances[asciiCode];
		maxWidth = std::max(maxWidth, tch.m_maxWidth);
		minWidth = std::min(minWidth, tch.m_minWidth);
	}
}
}

void TrainingData::estimateNormalDistributionParams()
{
	// computing the mean, max and min.
	for (auto& ch : m_charInstances)
	{
		auto samplesVec = getSamples(ch.first);
		double widthSum = 0;
		double heightSum = 0;
		ch.second.m_maxWidth = 0;
		ch.second.m_minWidth = 5000;

		for_each(samplesVec.begin(), samplesVec.end(), [&](const Character& ci)
		{
			widthSum += ci.m_loc.width;
			heightSum += ci.m_loc.height;

			ch.second.m_maxWidth = std::max(ch.second.m_maxWidth, ci.m_loc.width);
			ch.second.m_minWidth = std::min(ch.second.m_minWidth, ci.m_loc.width);
		});

		double meanWidth = (widthSum / samplesVec.size());
		double meanHeight = (heightSum / samplesVec.size());

		// computing the variance.
		double accumWidth = 0.0;
		for_each(begin(samplesVec), end(samplesVec), [&](const Character& ch)
		{
			accumWidth += (ch.m_loc.width - meanWidth) * (ch.m_loc.width - meanWidth);
		});

		double varWidth = accumWidth / (samplesVec.size() - 1);

		ch.second.m_widthMean = meanWidth;
		ch.second.m_widthStd = sqrt(varWidth);

		ch.second.m_heightMean = meanHeight;
	}
}

double TrainingData::getMinWidth(uchar asciiCode)
{
	if (asciiCode != '|')
	{
	return m_charInstances[asciiCode].m_minWidth;
}
	else
	{
		return 1;
	}
}

double TrainingData::getMaxWidth(uchar asciiCode)
{
	if (asciiCode != '|')
	{
	return m_charInstances[asciiCode].m_maxWidth;
}
	else
	{
		return 410;
	}
}

double TrainingData::getMeanWidth(uchar asciiCode)
{
	return m_charInstances[asciiCode].m_widthMean;
}

double TrainingData::getStdWidth(uchar asciiCode)
	{
	return m_charInstances[asciiCode].m_widthStd;
}
		
double TrainingData::getMeanHeight(uchar asciiCode)
{
	return m_charInstances[asciiCode].m_heightMean;
}

const Doc *TrainingData::getDocByName(string docName)
{
	Doc *ptr = nullptr;
	auto iter = m_file2trDoc.find(docName);
	if (iter != m_file2trDoc.end())
	{
		ptr = &m_trainingDocs[iter->second];
	}
	return ptr;
}

void TrainingData::writeQueriesAndDocsGTPfiles()
{
	ofstream queryFile;
	queryFile.open("queries.gtp");
	for (auto &doc : m_trainingDocs)
	{
		ofstream gtpFile;
		path docFileName = path(doc.m_pathImage).filename();
		string fileName = docFileName.stem().string() + ".gtp";
		gtpFile.open(fileName);
		for (auto &ch : doc.m_chars)
		{
			auto &loc = ch.m_loc;
			auto x_start = loc.x+1;
			auto y_start = loc.y+1;
			auto x_end = x_start + loc.width - 1;
			auto y_end = y_start + loc.height - 1;
			gtpFile << x_start << " " << y_start << " " << x_end << " " << y_end << " " << ch.m_asciiCode << endl;
			queryFile << docFileName.string() << " " << x_start << " " << y_start << " " << x_end << " " << y_end << " " << ch.m_asciiCode << endl;
		}
		gtpFile.close();
	}
	queryFile.close();
}

// TODO: iterate over the docs/word/line and write to each char where it came from.
void TrainingData::displayTrainingData()
{
	for (auto &ch : m_charInstances)
	{
		uchar asciiCode = ch.first;
		string pathString = m_params.m_pathData + char(asciiCode) + "_" + std::to_string(asciiCode) + "/";
		path dir(pathString);
		if (!exists(dir))
		if (boost::filesystem::create_directory(dir))
		{
			std::cout << "Success in creating " << dir.string() << "\n";
		}
		for (size_t i = 0; i < ch.second.m_instances.size(); ++i)
		{
			auto &ci = ch.second.m_instances[i];

			Doc &doc = m_trainingDocs[ci.m_docNum];
			Mat &imDoc = doc.m_origImage;
			Rect loc = ci.m_loc;

			auto x1 = loc.x;
			auto x2 = loc.x + loc.width;
			auto y1 = loc.y;
			auto y2 = loc.y + loc.height;
			Mat im = imDoc(Rect(x1, y1, x2 - x1, y2 - y1));

			imwrite(pathString + to_string(i)+".png", im);
		}
	}
}
