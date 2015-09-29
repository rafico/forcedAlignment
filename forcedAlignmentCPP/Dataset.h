#ifndef _MY_DATASET_H_
#define _MY_DATASET_H_

/************************************************************************
Project:  Text Alignment
Module:   Dataset Definitions
Purpose:  Defines the data structs of instance and label

*************************** INCLUDE FILES ******************************/
#include <cstdlib>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "CharClassifier.h"


class IntVector : public std::vector<int> {
public:
	unsigned int read(std::string &filename) {
		std::ifstream ifs(filename.c_str());
		// check input file stream
		if (!ifs.good()) {
			std::cerr << "Error: Unable to read IntVector from " << filename << std::endl;
			exit(-1);
		}
		// delete the vector
		clear();
		// read size from the stream
		int value;
		int num_values;
		if (ifs.good())
			ifs >> num_values;
		while (ifs.good() && num_values--) {
			ifs >> value;
			push_back(value);
		}
		ifs.close();
		return size();
	}

};

std::ostream& operator<< (std::ostream& os, const IntVector& v);

/***********************************************************************/


class CharSequence : public vector<uchar>
{
public:
	void from_string(const string &char_string);

public:
	static unsigned int m_num_chars;
};

std::ostream& operator<< (std::ostream& os, const CharSequence& y);

/***********************************************************************/

class StartTimeSequence : public std::vector<int>
{};

std::ostream& operator<< (std::ostream& os, const StartTimeSequence& y);

/***********************************************************************/

struct AnnotatedLine : Doc
{
	AnnotatedLine() : Doc()
	{}
	
	AnnotatedLine(string pathLine)
		: Doc(pathLine)
	{}

	double returnScore(uchar asciiCode, int cellStart, int cellEnd);

	struct scoresType
	{
		vector<Rect> m_locW;
		vector<double> m_scsW;
		HogSvmModel m_hs_model;
		
		// for debugging.
		Mat scoreVis;
	};

	unordered_map<uchar, scoresType> m_scores;

	CharSequence m_charSeq;
};

/***********************************************************************/

class Dataset
{
public:
	Dataset(CharClassifier& lm);
	void read(AnnotatedLine &x, StartTimeSequence &y);
	unsigned long size() { return m_start_times_file.size(); }

	void loadTrainingData();

private:

	struct Example
	{
		AnnotatedLine m_line;
		StartTimeSequence m_time_seq;
	};

	void parseFiles();
	void loadImageAndcomputeScores(AnnotatedLine &x);
	void verifyGTconsistency(AnnotatedLine & x);
	
	StringVector m_training_file_list;
	StringVector m_validation_file_list;

	StringVector m_transcription_file;
	StringVector m_start_times_file;
	int m_current_line;

	unordered_map<string, Example> m_examples;
	vector<string> m_lineIds;

	bool m_isParsed = false;
	Params &m_params;
	CharClassifier &m_lm;
};


#endif // _MY_DATASET_H_
