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
#include <map>
#include "Doc.h"
#include "commonTypes.h"
#include "Params.h"

#define MAX_LINE_SIZE 4096

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

class StringVector : public std::vector<std::string> {
public:

	unsigned int read(const std::string &filename) {
		std::ifstream ifs;
		char line[MAX_LINE_SIZE];
		ifs.open(filename.c_str());
		if (!ifs.is_open()) {
			std::cerr << "Unable to open file list:" << filename << std::endl;
			return 0;
		}
		while (!ifs.eof()) {
			ifs.getline(line, MAX_LINE_SIZE);
			if (strcmp(line, ""))
				push_back(std::string(line));
		}
		ifs.close();
		return size();
	}
};

/***********************************************************************/

class CharSequence : public vector<int>
{
public:
	void from_string(const string &char_string);

public:
	static unsigned int m_num_chars;
};

std::ostream& operator<< (std::ostream& os, const CharSequence& y);

/***********************************************************************/

class StartTimeSequence : public std::vector<int>
{
public:
	uint read(std::string &line);
};

std::ostream& operator<< (std::ostream& os, const StartTimeSequence& y);

/***********************************************************************/

struct AnnotatedLine : Doc
{
	AnnotatedLine() : Doc()
	{}
	
	AnnotatedLine(string pathLine)
		: Doc(pathLine)
	{}

	CharSequence m_char_seq;
};

/***********************************************************************/

class Dataset
{
public:
	Dataset(const Params& params);
	unsigned int read(AnnotatedLine &x, StartTimeSequence &y);
	unsigned long size() { return m_start_times_file.size(); }

private:

	struct Example
	{
		string lineId;
		AnnotatedLine m_line;
		StartTimeSequence m_time_seq;
	};

	void parseFiles();

	StringVector m_transcription_file;
	StringVector m_start_times_file;
	int m_current_line;

	vector<Example> m_examples;

	bool m_isParsed = false;
	const Params& m_params;
};


#endif // _MY_DATASET_H_
