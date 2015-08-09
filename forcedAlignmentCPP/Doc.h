#ifndef _H_DOC_H__
#define _H_DOC_H__

#include "commonTypes.h"
#include "CharInstance.h"

struct Doc
{
	Doc(Mat image, uint nChars, vector<CharInstance> && chars, uint H, uint W, string pathImage);

	Mat m_origImage;
	Mat m_image;
	Mat m_features;
	uint m_nChars;
	vector<CharInstance> m_chars;
	uint m_W;
	uint m_H;
	string m_pathImage;
	uint m_yIni;
	uint m_xIni;
	uint m_bH;
	uint m_bW;
};


#endif // !_H_DOC_H__
