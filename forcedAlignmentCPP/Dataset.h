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
#include "CharSequence.h"
#include "AnnotatedLine.h"

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

class StartTimeSequence : public std::vector<int>
{};

std::ostream& operator<< (std::ostream& os, const StartTimeSequence& y);

/***********************************************************************/

class Dataset
{
public:
	Dataset(string file_list, string start_times_file, bool accTrans = true);
	void read(AnnotatedLine &x, StartTimeSequence &y);
	void read(AnnotatedLine &x, StartTimeSequence &y, int lineNum);
	void read(AnnotatedLine &x, Mat &lineEnd, Mat &lineEndBin, StartTimeSequence &y, int lineNum);
	size_t size() { return m_lineIds.size(); }
	bool labels_given() { return m_read_labels; }

private:

	struct Example
	{
		AnnotatedLine m_line;
		StartTimeSequence m_time_seq;
	};

	void parseFiles();
	void loadStartTimes(Example &example, const string docName, int lineNum, const StartTimeSequence& startTimeAndEndTime);

	StringVector m_file_list;
	StringVector m_transcription_file;
	StringVector m_start_times_file;
	int m_current_file;
	int m_current_line;
	bool m_read_labels;
	bool m_accTrans;

	unordered_map<string, Example> m_examples;
	vector<string> m_lineIds;

	Params &m_params;
	TrainingData &m_trData;
	CharClassifier m_chClassifier;
};


#endif // _MY_DATASET_H_
