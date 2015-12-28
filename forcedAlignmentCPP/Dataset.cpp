/************************************************************************
Project:  Phoneme Alignment
Module:   Dataset Definition
Purpose:  Defines the data structs of the instances and the labels
Date:     25 Jan., 2005

SpeechUtterance --> Doc
PhonemeSequence --> CharSequence


**************************** INCLUDE FILES *****************************/
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <algorithm>
#include <set>
#include <cctype>
#include <opencv2/core.hpp>
#include <boost/filesystem.hpp>
#include "Dataset.h"
#include "HogUtils.h"
#include "AnnotatedLine.h"

using namespace boost::filesystem;

using namespace std;

/************************************************************************
Function:     operator << for StartTimeSequence

Description:  Write StartTimeSequence& vector to output stream
Inputs:       std::ostream&, const StartTimeSequence&
Output:       std::ostream&
Comments:     none.
***********************************************************************/
std::ostream& operator<< (std::ostream& os, const StartTimeSequence& y)
{
	for (uint i = 0; i < y.size(); i++)
		os << y[i] << " ";

	return os;
}

/************************************************************************
Function:     Dataset::Dataset

Description:  Constructor
Inputs:       std::string dataset_filename
Output:       void.
Comments:     none.
***********************************************************************/
Dataset::Dataset(string file_list, string start_times_file, bool accTrans /* = true */)
	: m_current_line(0),
	m_read_labels(false),
	m_accTrans(accTrans),
	m_params(Params::getInstance()), 
	m_trData(TrainingData::getInstance())
{
	// Read list of files into StringVector
	m_file_list.read(m_params.m_pathSets + file_list);

	// reading file content into StringVector
	m_transcription_file.read(m_params.m_pathTranscription);
	if (!start_times_file.empty())
	{
	m_start_times_file.read(m_params.m_pathStartTime);
		m_read_labels = true;
	}
		
	sort(begin(m_transcription_file), end(m_transcription_file));
	sort(begin(m_start_times_file), end(m_start_times_file));
	parseFiles();
}


/************************************************************************
Function:     Dataset::read

Description:  Read next instance and label
Inputs:       AnnotatedLine&, StartTimeSequence&
Output:       void.
Comments:     none.
***********************************************************************/
void Dataset::read(AnnotatedLine &x, StartTimeSequence &y, int lineNum)
{
	string lineId = m_lineIds[lineNum];
	
	auto iter = m_examples.find(lineId);
	if (iter == m_examples.end())
	{
		cerr << "Could not find line: " << lineId << endl;
	}
	
	x = iter->second.m_line;
	x.Init(x.m_pathImage, m_params.m_pathLineBinImages);
	y = iter->second.m_time_seq;

	computeScores(x);

	int x_shift = x.m_xIni;
	transform(y.begin(), y.end(), y.begin(), [x_shift](int startTime){return std::max(startTime - x_shift, 0); });	

	x.m_lineId = lineId;
}

void Dataset::read(AnnotatedLine &x, Mat &lineEnd, Mat &lineEndBin, StartTimeSequence &y, int lineNum)
{
	string lineId = m_lineIds[lineNum];

	auto iter = m_examples.find(lineId);
	if (iter == m_examples.end())
	{
		cerr << "Could not find line: " << lineId << endl;
	}

	x = iter->second.m_line;
	y = iter->second.m_time_seq;

	//string binImgPath =  + p.stem().string() + ".png";

	if (lineEnd.empty())
	{
		x.Init(x.m_pathImage, m_params.m_pathLineBinImages);
		computeScores(x);

		int x_shift = x.m_xIni;
		transform(y.begin(), y.end(), y.begin(), [x_shift](int startTime){return std::max(startTime - x_shift, 0); });
	}
	else
	{
		x.InitCombinedImg(x.m_pathImage, m_params.m_pathLineBinImages, lineEnd, lineEndBin);
		computeScores(x);

		int x_shift = x.m_xIni-lineEnd.cols;
		transform(y.begin(), y.end(), y.begin(), [x_shift](int startTime){return std::max(startTime - x_shift, 0); });
	}

	x.m_lineId = lineId;
}

void Dataset::read(AnnotatedLine &x, StartTimeSequence &y)
{
	read(x, y, m_current_line++);
}

std::ostream& operator<< (std::ostream& os, const IntVector& v)
{
	for_each(begin(v), end(v), [&](int val){os << val << " "; });
	return os;
}

void Dataset::parseFiles()
{
	for (auto fileName : m_file_list)
	{
		auto p = std::lower_bound(begin(m_transcription_file), end(m_transcription_file), fileName);
		string docName;
		string lineId;

		stringstream ssTranscription(*p++);
		ssTranscription >> lineId;
		docName = lineId.substr(0, lineId.find_last_of("-"));
		while (p != m_transcription_file.end() && docName.compare(fileName) == 0)
		{
			string transcriptionString;
			string transcriptionString2;
			ssTranscription >> transcriptionString >> transcriptionString2;
			
		Example example;
			if (m_accTrans)
			{ 
				example.m_line.m_charSeq.from_acc_trans(transcriptionString);
			}
			else
			{
				example.m_line.m_charSeq.from_in_acc_trans(transcriptionString2);
			}
			example.m_line.m_pathImage = m_params.m_pathLineImages + lineId + ".png";

			m_examples.insert({ lineId, example });
			m_lineIds.push_back(lineId);

			ssTranscription.str(*p++);
			ssTranscription.clear(); // Clear state flags.

			ssTranscription >> lineId;
			docName = lineId.substr(0, lineId.find_last_of("-"));
		}
	}
	if (m_read_labels)
	{
		for (auto lineId : m_lineIds)
		{
			auto p = std::lower_bound(begin(m_start_times_file), end(m_start_times_file), lineId);
			if (p == m_start_times_file.end())
			{
				cerr << "Could not find line: " << lineId << " in start times file." << endl;
			}
			stringstream ssStartTime(*p);
			string dummyLineId;
			ssStartTime >> dummyLineId;
		StartTimeSequence startTimeAndEndTime;
		startTimeAndEndTime.assign((istream_iterator<int>(ssStartTime)), (istream_iterator<int>()));

		size_t found = lineId.find_last_of("-");
		string docName = lineId.substr(0, found);
		int lineNum = stoi(lineId.substr(found + 1));

			Example &example = m_examples[lineId];
			loadStartTimes(example, docName, lineNum, startTimeAndEndTime);
		}
			}
		}


void Dataset::loadStartTimes(Example &example, string docName, int lineNum, const StartTimeSequence& startTimeAndEndTime)
{
		const Doc *doc = m_trData.getDocByName(docName);
		if (doc != nullptr)
		{
			// verify consistency with our GT.
			const Line &line = doc->m_lines[lineNum - 1];
			size_t ALchIdx = 0;
			size_t timeSeqIdx = 0;
			for (size_t wordIdx = 0; wordIdx < line.m_wordIndices.size(); ++wordIdx)
			{
				const Word &word = doc->m_words[line.m_wordIndices[wordIdx]];
				for (auto &chIdx : word.m_charIndices)
				{
					auto ch = doc->m_chars[chIdx].m_asciiCode;
					auto ALch = example.m_line.m_charSeq[ALchIdx];
					if (ALch == '|')
					{
						int prevStartTime = (timeSeqIdx == 0) ? 0 : startTimeAndEndTime[2 * (timeSeqIdx - 1)];
						int endTime = (timeSeqIdx == 0) ? 0 : (prevStartTime + startTimeAndEndTime[2 * timeSeqIdx - 1] + 1);
						int nextStartTime = startTimeAndEndTime[2 * timeSeqIdx];
						endTime = std::min(endTime, nextStartTime - 1);
						example.m_time_seq.push_back(endTime);
						ALch = example.m_line.m_charSeq[++ALchIdx];
					}
					if (ch != ALch)
					{
						cerr << "Inconsistency in Doc: " << docName << " Line #: " << lineNum << " Word #: " << wordIdx+1 << endl;
						cerr << "Expected " << ch << " but got " << ALch << endl;
					}
					else if (ALch != '|')
					{
						int startTime = startTimeAndEndTime[2 * timeSeqIdx];
						example.m_time_seq.push_back(startTime);
					}
					++ALchIdx;
					++timeSeqIdx;
				}
			}

			int endTime = startTimeAndEndTime[2 * (timeSeqIdx - 1)] + startTimeAndEndTime[2 * timeSeqIdx - 1] + 1;
			example.m_time_seq.push_back(endTime);
	}
}

void Dataset::computeScores(AnnotatedLine &x, const CharSequence *charSeq /*= nullptr */)
{
	uint sbin = m_params.m_sbin;

	// Is it the first time we compute scores ?
	if (nullptr == charSeq)
	{
	x.computeFeatures(sbin);
		x.computeFixedScores();
	}

	const CharSequence &cq = (nullptr == charSeq) ? x.m_charSeq : (*charSeq);

	for (auto asciiCode : cq)
	{

		set<uchar> asciiCodes = {asciiCode, m_accTrans? asciiCode:(uchar)std::toupper(asciiCode)};

		for (auto asciiCode_ : asciiCodes)
	{
		AnnotatedLine::scoresType scores;
			scores.m_scoreVals = Mat::ones(1, x.m_W, CV_64F)*-1;

			if (asciiCode_ != '|')
		{
		// Computing scores for this model over the line.
				if (x.m_scores.find(asciiCode_) == x.m_scores.end())
		{
					scores.m_hs_model = m_chClassifier.learnModel(asciiCode_);

					if (scores.m_hs_model.isInitialized())
					{
			vector<Rect> locW;
			vector<double> scsW;
			HogUtils::getWindows(x, scores.m_hs_model, scsW, locW, m_params.m_step, sbin, false);

			// computing HOG scores.
				double *ptr = scores.m_scoreVals.ptr<double>(0);
			for (size_t i = 0; i < locW.size(); ++i)
			{
				int xInd = locW[i].x*sbin;
				ptr[xInd] = std::max(ptr[xInd], 10*scsW[i]);
			}

				for (auto idx = 0; idx < scores.m_scoreVals.cols; ++idx)
			{
				ptr[idx] = ptr[idx - (idx % sbin)];
			}
			}
					else
					{
						clog << "No model for " << asciiCode_ << ", trying to continue." << endl;
					}
				}
				x.m_scores.insert({ asciiCode_, move(scores) });
			}
		}
	}
}

