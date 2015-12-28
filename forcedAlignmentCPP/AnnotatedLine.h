#ifndef _H_ANNOTATED_LINE_H__
#define _H_ANNOTATED_LINE_H__

#include "Doc.h"
#include "HogSvmModel.h"
#include "CharSequence.h"
#include <opencv2/core.hpp>

struct AnnotatedLine : Doc
{
	AnnotatedLine() : Doc()
	{}

	AnnotatedLine(string pathLine)
		: Doc(pathLine)
	{}

	struct scoresType
	{
		HogSvmModel m_hs_model;
		Mat m_scoreVals;
	};

	void InitCombinedImg(string pathImage, string binPath, Mat lineEnd, Mat lineEndBin);
	void computeFixedScores();

	unordered_map<uchar, scoresType> m_scores;
	Mat m_fixedScores;

	CharSequence m_charSeq;
	string m_lineId;
};

#endif