#ifndef _H_DOC_H__
#define _H_DOC_H__

#include "commonTypes.h"
#include "CharInstance.h"

// class for holding both document images and text line images.

struct Doc
{
	Doc();
	Doc(string pathImage, vector<CharInstance> && chars = vector<CharInstance>());
	void Init(string pathImage);

	void resizeDoc(uint sbin);
	void computeFeatures(uint sbin);

	Mat m_origImage;
	Mat m_image;
	string m_pathImage;
	Mat m_features;
	vector<CharInstance> m_chars;
	uint m_nChars;
	uint m_H;	
	uint m_W;
	uint m_yIni;
	uint m_xIni;
	int m_bH;
	int m_bW;
};

#endif // !_H_DOC_H__
