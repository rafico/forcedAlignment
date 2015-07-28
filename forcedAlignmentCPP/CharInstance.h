#ifndef _H_MODEL_INSTANCE_H__
#define _H_MODEL_INSTANCE_H__

#include "commonTypes.h"
/*
each 'csv' file represents a matrix of size numOfCharacter x 6, and each row in the following format

Threshold  Ascii_code X  Y W H

(X, Y) is the top left coordinate of the character's bounding box
W, H - are the width and height coordinates of the bounding box
*/
struct CharInstance
{
	CharInstance(const string &pathIm, uint globalIdx, const string &csv_line);

	string m_pathIm;
	uint m_globalIdx;

	uchar m_threshold;
	uchar m_asciiCode;
	Rect m_loc;
	uint m_classNum;
};
#endif // !_H_MODEL_INSTANCE_H__
