#ifndef H_QUERY_H

#include "commonTypes.h"

struct Query
{
	Query(string pathIm, Rect loc, uchar text, uint H, uint W)
		:
		m_pathIm(pathIm),
		m_loc(loc),
		m_text(text),
		m_H(H),
		m_W(W)
	{}
	
	string m_pathIm;
	Rect m_loc;
	uchar m_text;
	uint m_H;
	uint m_W;
};

#endif // !H_QUERY_H
