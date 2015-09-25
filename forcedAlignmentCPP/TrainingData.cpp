#include <numeric>
#include "TrainingData.h"
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace boost::filesystem;
using namespace std;

TrainingData::TrainingData(Params& params)
	: m_params(params), m_numRelevantWordsByClass(UCHAR_MAX + 1, 0)
{
	std::clog << "* Loading documents *" << endl;

	path dir_path(m_params.m_pathGT);
	if (!exists(dir_path))
	{
		cerr << "Error: Directory " << m_params.m_pathGT << " does not exist" << std::endl;
		return;
	}

	StringVector training_file_list;
	training_file_list.read(params.m_pathTrainingFiles);

	for (const auto &fileName : training_file_list)
	{
		cout << "Loading " << fileName << endl;
		string imgFileName = m_params.m_pathImages + path(fileName).filename().stem().string() + ".jpg";
		Doc doc(imgFileName);
		doc.loadXml(m_params.m_pathGT + fileName);
		m_trainingDocs.push_back(Doc(doc));
	}

	combineChars();
	//displayTrainingData();
	//writeQueriesAndDocsGTPfiles();
}

void TrainingData::combineChars()
{
	uint classNum = 0;
	for (size_t docNum = 0; docNum < m_trainingDocs.size(); ++docNum)
	{
		for (auto &ch : m_trainingDocs[docNum].m_chars)
		{
			ch.m_docNum = docNum;
			auto iter = m_charInstances.find(ch.m_asciiCode);
			if (iter != m_charInstances.end())
			{
				classNum = iter->second.m_class;
				iter->second.m_instances.push_back(ch);
			}
			else
			{
				TrainingCharsHelper tch;
				classNum = tch.m_class = uint(m_charInstances.size());
				tch.m_instances.push_back(ch);
				m_charInstances.insert({ ch.m_asciiCode, tch });
			}
			++m_numRelevantWordsByClass[classNum];
		}
	}

	auto iter = find_if(m_numRelevantWordsByClass.begin(), m_numRelevantWordsByClass.end(), [](uint cnt){return cnt == 0; });
	auto numClasses = distance(m_numRelevantWordsByClass.begin(), iter);
	m_numRelevantWordsByClass.resize(numClasses);

	size_t numDocs = m_trainingDocs.size();
	m_relevantBoxesByClass.resize(numDocs*numClasses);
	for (size_t i = 0; i < numDocs; ++i)
	{
		const auto& chars = m_trainingDocs[i].m_chars;
		for (auto &ch : chars)
		{
			uint classNum = m_charInstances[ch.m_asciiCode].m_class;
			Rect loc = ch.m_loc;
			size_t idx = i*numClasses + classNum;
			m_relevantBoxesByClass[idx].push_back(loc);
		}
	}
	m_params.m_numNWords = ceil((double)m_params.m_numNWords / numDocs)*numDocs;
}

const vector<Character>& TrainingData::getSamples(uint asciiCode)
{
	auto& tch = m_charInstances[asciiCode];
	return tch.m_instances;
}

void TrainingData::getExtermalWidths(int &maxWidth, int& minWidth)
{
	maxWidth = m_globalMaxWidth;
	minWidth = m_globalMinWidth;
}


void TrainingData::computeNormalDistributionParams()
{
	m_globalMaxWidth = 0;
	m_globalMinWidth = 5000;

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

		m_globalMaxWidth = std::max(m_globalMaxWidth, ch.second.m_maxWidth);
		m_globalMinWidth = std::min(m_globalMinWidth, ch.second.m_minWidth);
		
		double meanWidth = (widthSum / samplesVec.size());
		double meanHeight = (heightSum / samplesVec.size());

		// computing the variance.
		double accumWidth = 0.0;
		double accumHeight = 0.0;
		for_each(begin(samplesVec), end(samplesVec), [&](const Character& ch)
		{
			accumWidth += (ch.m_loc.width - meanWidth) * (ch.m_loc.width - meanWidth);
			accumHeight += (ch.m_loc.height - meanHeight) * (ch.m_loc.height - meanHeight);
		});

		double varWidth = accumWidth / (samplesVec.size() - 1);
		double varHeight = accumHeight / (samplesVec.size() - 1);

		ch.second.m_widthMean = meanWidth;
		ch.second.m_widthStd = sqrt(varWidth);

		ch.second.m_heightMean = meanHeight;
		ch.second.m_heightStd = sqrt(varHeight);
	}
}

double TrainingData::getMeanWidth(uint asciiCode)
{
	return m_charInstances[asciiCode].m_widthMean;
}

double TrainingData::getMeanHeight(uint asciiCode)
{
	return m_charInstances[asciiCode].m_heightMean;
}

void TrainingData::load_char_stats(charStatType &meanCont, charStatType& stdCont) const
{
	for (auto& ch : m_charInstances)
	{
		uchar asciiCode = ch.first;
		double mean = ch.second.m_widthMean;
		double std = ch.second.m_widthStd;
		
		meanCont.insert({asciiCode, mean});
		stdCont.insert({asciiCode, std});
	}
}

uint TrainingData::getCharClass(uchar asciiCode)
{
	return m_charInstances[asciiCode].m_class;
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
