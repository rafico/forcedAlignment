#ifndef H_QUERY_H

#include "commonTypes.h"

struct Query
{
	Query(const string &pathImages, const string &csv_line);
	
	uint getText() { return m_text; }
	
	uint getClassNum() { return m_class; }
	void setClassNum(uint classNum) { m_class = classNum; }

	string m_pathIm;
	Rect m_loc;
	uchar m_text;
	uint m_H;
	uint m_W;
	uint m_class;
};

#endif // !H_QUERY_H
