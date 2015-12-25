#ifndef _CHAR_SEQUENCE_H_
#define _CHAR_SEQUENCE_H_

#include "commonTypes.h"

class CharSequence : public vector<uchar>
{
public:
	void from_acc_trans(const string &char_string);
	void from_in_acc_trans(const string &char_string);

public:
	static unsigned int m_num_chars;
};

std::ostream& operator<< (std::ostream& os, const CharSequence& y);

#endif