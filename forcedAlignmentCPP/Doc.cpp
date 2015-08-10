#include "Doc.h"


Doc::Doc(Mat image, uint nChars, vector<CharInstance> && chars, uint H, uint W, string pathImage)
	: m_origImage(image),
	m_nChars(nChars),
	m_chars(chars),
	m_H(H),
	m_W(W),
	m_pathImage(pathImage)
{}