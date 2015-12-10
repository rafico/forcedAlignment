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
void CharSequence::from_acc_trans(const string &transcript)
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
Function:     PhonemeSequence::from_string

Description:  Read PhonemeSequence from a string
Inputs:       string &phoneme_string
Output:       void.
Comments:     none.
***********************************************************************/
void CharSequence::from_in_acc_trans(const string &transcript)
{
	string buffer;
	push_back('|');
	stringstream ss(transcript);
	while (std::getline(ss, buffer, '|'))
	{
		if (buffer.compare("BREAK"))
		{
			insert(this->end(), buffer.begin(), buffer.end());
			push_back('|');
		}
	}
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

void AnnotatedLine::InitCombinedImg(string pathImage, string binPath, Mat lineEnd, Mat lineEndBin)
{
	m_featuresComputed = false;
	Mat origImg = cv::imread(pathImage);
	
	m_pathImage = pathImage;
	if (!origImg.data)
	{
		std::cerr << "Could not open or find the image " << m_pathImage << std::endl;
	}
	
	m_origImage = Mat::zeros(std::max(origImg.rows, lineEnd.rows), origImg.cols + lineEnd.cols, CV_8UC3);
	lineEnd.copyTo(m_origImage(Range(0, lineEnd.rows), Range(0, lineEnd.cols)));
	origImg.copyTo(m_origImage(Range(0, origImg.rows), Range(lineEnd.cols, m_origImage.cols)));

	path p(m_pathImage);
	string binImgPath = binPath + p.stem().string() + ".png";
	Mat origBin = cv::imread(binImgPath, CV_LOAD_IMAGE_GRAYSCALE);

	m_bin = Mat::zeros(std::max(origImg.rows, lineEndBin.rows), origImg.cols + lineEndBin.cols, CV_8UC1);
	lineEndBin.copyTo(m_bin(Range(0, lineEndBin.rows), Range(0, lineEndBin.cols)));
	origBin.copyTo(m_bin(Range(0, origBin.rows), Range(lineEndBin.cols, m_bin.cols)));

	m_H = m_origImage.rows;
	m_W = m_origImage.cols;
}

void Dataset::computeScores(AnnotatedLine &x, const CharSequence *charSeq /*= nullptr */)
{
	uint sbin = m_params.m_sbin;

	// Is it the first time we compute scores ?
	if (nullptr == charSeq)
	{
	x.computeFeatures(sbin);
		computeFixedScores(x);
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

void Dataset::computeFixedScores(AnnotatedLine &x)
{
	// row 0 - projection profile (start char)
	// row 1 - connected components (start char)
	// row 2 - integral projection profile.
	// row 3 - projection profile (end char)
	// row 4 - connected components (end char)

	x.m_fixedScores = Mat::zeros(5, x.m_W, CV_64F);

			const int intervalSz = 5;

			// computing Projection Profile scores.

			Mat pp;
	reduce(x.m_bin, pp, 0, CV_REDUCE_SUM, CV_64F);
	Mat ppMat = x.m_fixedScores.rowRange(Range(0, 1));
	Mat ppEndMat = x.m_fixedScores.rowRange(Range(3, 4));
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
			ppMat.colRange(std::max(i - intervalSz + 2, 0), std::min(int(i + intervalSz + 1), pp.cols)) = 1;
	}
		if (localMin && currentVal < pp.at<double>(std::max(i - 1, 0)))
		{
			ppEndMat.colRange(std::max(i - intervalSz + 2, 0), std::min(int(i + intervalSz + 1), pp.cols)) = 1;
		}
}

			// computing score using CC's
	Mat ccMat = x.m_fixedScores.rowRange(Range(1, 2));
	Mat ccEndMat = x.m_fixedScores.rowRange(Range(4, 5));
			Mat labels;
	connectedComponents(x.m_bin.t(), labels);
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
		diff.clear();

		set_difference(current.begin(), current.end(), next.begin(), next.end(), inserter(diff, diff.begin()));
		if (!diff.empty())
		{
			ccEndMat.colRange(std::max(colIdx - intervalSz + 1, 0), std::min(int(colIdx + intervalSz), labels.cols)) = 1;
		}
		diff.clear();

				current = move(next);
				next.clear(); 
			}

	// computing integral Projection Profile
	Mat intPPMat;
	integral(pp, intPPMat);
	intPPMat /= 255;
	intPPMat.rowRange(1, 2).colRange(1, pp.cols + 1).copyTo(x.m_fixedScores.rowRange(Range(2, 3)));
}
