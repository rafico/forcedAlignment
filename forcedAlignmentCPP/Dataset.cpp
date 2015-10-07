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
	push_back('|');
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
				if (transcript[i] == '|')
				{
					push_back('|');
				}
				buffer.clear();
				break;
			default:
				buffer.push_back(ch);
		}
	}
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
	push_back('|');
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
Dataset::Dataset()
	: m_current_line(0), 
	m_params(Params::getInstance()), 
	m_trData(TrainingData::getInstance())
{
	// Read list of files into StringVector
	m_training_file_list.read(m_params.m_pathTrainingFiles);
	m_validation_file_list.read(m_params.m_pathValidationFiles);

	// reading file content into StringVector
	m_transcription_file.read(m_params.m_pathTranscription);
	m_start_times_file.read(m_params.m_pathStartTime);
	parseFiles();
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
	transform(y.begin(), y.end(), y.begin(), [x_shift](int startTime){return std::max(startTime - x_shift, 0); });	
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
	for (size_t index = 0; index < m_start_times_file.size(); ++index)
	{
		stringstream ssStartTime(m_start_times_file[index]);
		stringstream ssTranscription(m_transcription_file[index]);

		string lineId, lindIdTranscription;
		ssStartTime >> lineId;
		ssTranscription >> lindIdTranscription;
		if (lineId.compare(lindIdTranscription))
		{
			cerr << "Transcript file and start_time file are not synchronized.\n";
		}
		Example example;

		StartTimeSequence startTimeAndEndTime;
		startTimeAndEndTime.assign((istream_iterator<int>(ssStartTime)), (istream_iterator<int>()));

		string transcriptionString;
		ssTranscription >> transcriptionString;
		example.m_line.m_charSeq.from_string(transcriptionString);

		example.m_line.m_pathImage = m_params.m_pathLineImages + lineId + ".png";

		size_t found = lineId.find_last_of("-");
		string docName = lineId.substr(0, found);
		int lineNum = stoi(lineId.substr(found + 1));

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

		m_examples.insert({lineId, example});
		m_lineIds.push_back(lineId);
	}
	}
}

void Dataset::loadImageAndcomputeScores(AnnotatedLine &x)
{
	uint sbin = m_params.m_sbin;
	x.Init(x.m_pathImage);
	x.computeFeatures(sbin);

	Mat fixedScores;
	computeFixedScores(x, fixedScores);

	for (auto asciiCode : x.m_charSeq)
	{
		AnnotatedLine::scoresType scores;
		scores.m_scoreVals = Mat::zeros(4, x.m_W, CV_64F);

		if (asciiCode != '|')
		{
		// Computing scores for this model over the line.
		if (x.m_scores.find(asciiCode) == x.m_scores.end())
		{
			scores.m_hs_model = m_chClassifier.learnModel(asciiCode);

			vector<Rect> locW;
			vector<double> scsW;
			HogUtils::getWindows(x, scores.m_hs_model, scsW, locW, m_params.m_step, sbin, false);

			// computing HOG scores.
				double *ptr = scores.m_scoreVals.ptr<double>(0);
				scores.m_scoreVals.rowRange(0, 1) = -1;
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
		}
			fixedScores.copyTo(scores.m_scoreVals.rowRange(1, 4));
			x.m_scores.insert({ asciiCode, move(scores) });
	}
}


void Dataset::computeFixedScores(AnnotatedLine &x, Mat &scores)
{
	scores = Mat::zeros(3, x.m_W, CV_64F);

			const int intervalSz = 5;

			// computing Projection Profile scores.
			path p(x.m_pathImage);
			string binImgPath = m_params.m_pathLineBinImages + p.stem().string() + ".png";
			Mat OrigBin = cv::imread(binImgPath, CV_LOAD_IMAGE_GRAYSCALE);
			Mat bin = OrigBin(Range(x.m_yIni, x.m_image.rows + x.m_yIni), Range(x.m_xIni, x.m_image.cols + x.m_xIni));
			Mat pp;
			reduce(bin, pp, 0, CV_REDUCE_SUM, CV_64F);
	Mat ppMat = scores.rowRange(Range(0, 1));
			for (int i = 0; i < pp.cols; ++i)
			{
				bool localMin = true;
				double currentVal = pp.at<double>(i);
				for (int offset = -intervalSz; (offset <= intervalSz) && localMin; ++offset)
				{
					double neighbourIdx = std::min(std::max(i+offset,0),pp.cols);
					localMin &= currentVal <= pp.at<double>(neighbourIdx);
		}
				if (localMin && currentVal < pp.at<double>(std::min(i+1, pp.cols)))
				{
			ppMat.colRange(std::max(i - intervalSz + 1, 0), std::min(int(i + intervalSz), pp.cols)) = 1;
	}
}

			// computing score using CC's
	Mat ccMat = scores.rowRange(Range(1, 2));
			Mat labels;
			connectedComponents(bin.t(), labels);
			transpose(labels, labels);

			set<int> current;
			set<int> next;
			set<int> diff;

	for (auto rowIdx = 0; rowIdx < labels.rows; ++rowIdx)
{
				current.insert(labels.at<int>(rowIdx, 0));
			}

	for (auto colIdx = 1; colIdx < labels.cols - 1; ++colIdx)
	{
		for (auto rowIdx = 0; rowIdx < labels.rows; ++rowIdx)
		{
					next.insert(labels.at<int>(rowIdx, colIdx));
				}

				set_difference(next.begin(), next.end(), current.begin(), current.end(), inserter(diff, diff.begin()));
				if (!diff.empty())
			{
			ccMat.colRange(std::max(colIdx - intervalSz + 1, 0), std::min(int(colIdx + intervalSz), labels.cols)) = 1;
				}
				current = move(next);
				next.clear(); 
				diff.clear();
			}

	// computing integral Projection Profile
	Mat intPPMat;
	integral(pp, intPPMat);
	intPPMat /= 255;
	intPPMat.rowRange(1,2).colRange(1,pp.cols+1).copyTo(scores.rowRange(Range(2, 3)));
}
