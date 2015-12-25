#include <iostream>
#include <sstream>
#include <fstream>
#include "CharSequence.h"

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
	std::stringstream ss(transcript);
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