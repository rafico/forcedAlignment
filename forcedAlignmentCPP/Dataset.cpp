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
#include "Dataset.h"

using namespace std;

// PhonemeSequence static memebers definitions
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
	for (size_t i = 0; i < transcript.size(); ++i)
	{
		switch (transcript[i])
		{
			case '-':
				 continue;
			case '|':
				push_back(' ');
			default:
				push_back(transcript[i]);
		}
	}
}


/************************************************************************
Function:     operator << for PhonemeSequence

Description:  Write PhonemeSequence& vector to output stream
Inputs:       std::ostream&, const StringVector&
Output:       std::ostream&
Comments:     none.
***********************************************************************/
std::ostream& operator<< (std::ostream& os, const CharSequence& y)
{
	for (uint i = 0; i < y.size(); i++)
		os << y[i] << " ";

	return os;
}

uint StartTimeSequence::read(std::string &line)
{
	std::stringstream ss;
	/*
	std::ifstream ifs(filename.c_str());
	// check input file stream
	if (!ifs.good()) {
		std::cerr << "Error: Unable to read StartTimeSequence from " << filename << std::endl;
		exit(-1);
	}
	// delete the vector
	clear();
	// read size from the stream
	while (ifs.good()) {
		std::string value;
		ifs >> value;
		if (value == "") break;
		push_back(int(std::atoi(value.c_str())));
	}
	ifs.close();
	*/
	return size();
}

/************************************************************************
Function:     operator << for PhonemeSequence

Description:  Write PhonemeSequence& vector to output stream
Inputs:       std::ostream&, const StringVector&
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
Dataset::Dataset(const Params& params)
	: m_params(params), m_current_line(0)
{
	// Read list of files into StringVector
	m_transcription_file.read(m_params.m_pathTranscription);
	m_start_times_file.read(m_params.m_pathStartTime);
}


/************************************************************************
Function:     Dataset::read

Description:  Read next instance and label
Inputs:       SpeechUtterance&
StartTimeSequence&
Output:       void.
Comments:     none.
***********************************************************************/
uint Dataset::read(AnnotatedLine &x, StartTimeSequence &y)
{
	if (!m_isParsed)
	{
		parseFiles();
	}

	string start_time_seq_line = m_start_times_file[m_current_line];
	string trascript_line = m_transcription_file[m_current_line];

	string lineId;

	return 0;
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
		string temp;
		ssTranscription >> temp;
		if (!lineId.compare(temp))
		{
			cerr << "Transcript file and start_time file are not synchronized.";
		}
	
		Example example;
		example.m_time_seq.assign((istream_iterator<int>(ssStartTime)), (istream_iterator<int>()));
		
		ssTranscription >> temp;
		example.m_line.m_char_seq.from_string(temp);
		
		int x = 2;
	}
	m_isParsed = true;
}