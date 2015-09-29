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
#include <opencv2/core.hpp>
#include <boost/filesystem.hpp>
#include "Dataset.h"
#include "HogUtils.h"

using namespace boost::filesystem;

using namespace std;

// CharSequence static member definitions
unsigned int CharSequence::m_num_chars;


/************************************************************************
Function:     PhonemeSequence::from_string

Description:  Read PhonemeSequence from a string
Inputs:       string &phoneme_string
Output:       void.
Comments:     none.
***********************************************************************/
void CharSequence::from_string(const string &transcript)
{
	string buffer;
	for (size_t i = 0; i < transcript.size(); ++i)
	{
		char ch = transcript[i];
		switch (ch)
		{
			case '-':
			case '|':
				if (buffer == "et")
				{
					push_back('&');
				}
				else if (buffer == "pt")
				{
					push_back('.');
				}
				else
				{
					push_back(buffer[0]);
				}
				// Currently we ignore the space between consecutive words.
				/*
				if (transcript[i] == '|')
				{
					push_back('|');
				}
				*/
				buffer.clear();
				break;
			default:
				buffer.push_back(ch);
		}
	}
	push_back(buffer[0]);
}


/************************************************************************
Function:     operator << for CharSequence

Description:  Write CharSequence& vector to output stream
Inputs:       std::ostream&, const CharSequence&
Output:       std::ostream&
Comments:     none.
***********************************************************************/
std::ostream& operator<< (std::ostream& os, const CharSequence& y)
{
	for (uint i = 0; i < y.size(); i++)
		os << y[i] << " ";

	return os;
}

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
Dataset::Dataset(CharClassifier& lm)
	: m_current_line(0), m_params(Params::getInstance()), m_lm(lm)
{
	// Read list of files into StringVector
	m_training_file_list.read(m_params.m_pathTrainingFiles);
	m_validation_file_list.read(m_params.m_pathValidationFiles);

	// reading file content into StringVector
	m_transcription_file.read(m_params.m_pathTranscription);
	m_start_times_file.read(m_params.m_pathStartTime);
}


/************************************************************************
Function:     Dataset::read

Description:  Read next instance and label
Inputs:       AnnotatedLine&, StartTimeSequence&
Output:       void.
Comments:     none.
***********************************************************************/
void Dataset::read(AnnotatedLine &x, StartTimeSequence &y)
{
	if (!m_isParsed)
	{
		parseFiles();
	}

	string lineId = m_lineIds[m_current_line++];
	
	auto iter = m_examples.find(lineId);
	if (iter == m_examples.end())
	{
		cerr << "Could not find line: " << lineId << endl;
	}
	
	x = iter->second.m_line;
	y = iter->second.m_time_seq;

	loadImageAndcomputeScores(x);

	int x_shift = x.m_xIni;
	uint sbin = m_params.m_sbin;
	transform(y.begin(), y.end(), y.begin(), [x_shift, sbin](int startTime){return floor(((double)startTime - x_shift) / sbin); });	
}

double AnnotatedLine::returnScore(uchar asciiCode, int startCell, int endCell)
{
	const HogSvmModel& hs_model = m_scores[asciiCode].m_hs_model;
	const vector<Rect> &locW = m_scores[asciiCode].m_locW;
	const vector<double> &scsW = m_scores[asciiCode].m_scsW;
	const Mat& vis = m_scores[asciiCode].scoreVis;

	double max_val = MISPAR_KATAN_MEOD;

	int _endCell = min(max(endCell, startCell + hs_model.m_bW), m_bW);

	Mat debug_im;

	if (startCell >= 0)
	{
		debug_im = m_image.colRange(startCell * 6, _endCell * 6);

		auto iter = find_if(locW.begin(), locW.end(), [startCell](const Rect& rect){return rect.x == startCell; });
		if (iter != locW.end())
		{
			size_t index = distance(locW.begin(), iter);
			for (; index < locW.size() && 
				locW[index].x + hs_model.m_bW <= _endCell;
				++index)
			{
				max_val = std::max(max_val, scsW[index]);
			}
		}
	}
	
	return max_val;
}

std::ostream& operator<< (std::ostream& os, const IntVector& v)
{
	IntVector::const_iterator iter = v.begin();
	IntVector::const_iterator end = v.end();

	while (iter < end) {
		os << *iter << " ";
		++iter;
	}
	return os;
}

void Dataset::parseFiles()
{
	string lineId;
	for (size_t index = 0; index < m_start_times_file.size(); ++index)
	{
		stringstream ssStartTime(m_start_times_file[index]);
		stringstream ssTranscription(m_transcription_file[index]);

		ssStartTime >> lineId;
		string lindIdTranscription;
		ssTranscription >> lindIdTranscription;
		if (lineId.compare(lindIdTranscription))
		{
			cerr << "Transcript file and start_time file are not synchronized.\n";
		}
		Example example;

		example.m_time_seq.assign((istream_iterator<int>(ssStartTime)), (istream_iterator<int>()));

		string transcriptionString;
		ssTranscription >> transcriptionString;
		example.m_line.m_charSeq.from_string(transcriptionString);

		example.m_line.m_pathImage = m_params.m_pathLineImages + lineId + ".png";

		m_examples.insert({lineId, example});
		m_lineIds.push_back(lineId);
	}
	m_isParsed = true;
}

void Dataset::loadImageAndcomputeScores(AnnotatedLine &x)
{
	uint sbin = m_params.m_sbin;
	x.Init(x.m_pathImage);
	x.computeFeatures(sbin);

	verifyGTconsistency(x);

	for (auto asciiCode : x.m_charSeq)
	{
		// Computing scores for this model over the line, then sorting the scores according to the abscissa (x coordinate).
		if (x.m_scores.find(asciiCode) == x.m_scores.end())
		{
			AnnotatedLine::scoresType scores;
			scores.m_hs_model = m_lm.learnModel(asciiCode);

			vector<Rect> locW;
			vector<double> scsW;
			HogUtils::getWindows(x, scores.m_hs_model, scsW, locW, m_params.m_step, m_params.m_sbin, false);

			vector<int> x_indices;
			transform(locW.begin(), locW.end(), std::back_inserter(x_indices), [](Rect& locW) { return locW.x; });
			Mat scoresIdx;
			sortIdx(Mat(x_indices), scoresIdx, cv::SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

			scores.m_locW.resize(x_indices.size());
			scores.m_scsW.resize(x_indices.size());

			for (size_t i = 0; i < x_indices.size(); ++i)
			{
				int index = scoresIdx.at<int>(i);
				scores.m_locW[i] = locW[index];
				scores.m_scsW[i] = scsW[index];
			}

			scores.scoreVis = Mat::zeros(x.m_bH, x.m_bW, CV_64F);
			for (size_t i = 0; i < locW.size(); ++i)
			{
				int xInd = locW[i].x;
				int yInd = locW[i].y;
				scores.scoreVis.at<double>(yInd, xInd) = scsW[i];
			}

			x.m_scores.insert({ asciiCode, move(scores) });
		}
	}
}

void Dataset::verifyGTconsistency(AnnotatedLine & x)
{
	// TODO: rewrite this function.
	path linePath(x.m_pathImage);

	auto temp = linePath.stem().string();
	size_t found = temp.find_last_of("-");
	string docName = temp.substr(0, found);
	int lineNum = stoi(temp.substr(found + 1));

	const Doc &doc = m_lm.getDocByName(docName);
	const Line &line = doc.m_lines[lineNum-1];
	size_t ALchIdx = 0;
	for (size_t wordIdx = 0; wordIdx < line.m_wordIndices.size(); ++ wordIdx)
	{
		const Word &word = doc.m_words[line.m_wordIndices[wordIdx]];
		for (auto &chIdx : word.m_charIndices)
		{
			auto ch = doc.m_chars[chIdx].m_asciiCode;
			auto ALch = x.m_charSeq[ALchIdx++];
			if (ch != ALch)
			{
				cerr << "Inconsistency in Doc: " << docName << " Line #: " << lineNum << " Word #: " << wordIdx << endl;
				cerr << "Expected " << ch << " but got " << ALch << endl;
			}
		}
	}
}
