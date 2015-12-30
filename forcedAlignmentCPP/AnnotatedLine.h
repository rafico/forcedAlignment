#ifndef _H_ANNOTATED_LINE_H__
#define _H_ANNOTATED_LINE_H__

#include <opencv2/core.hpp>
#include "Doc.h"
#include "HogSvmModel.h"
#include "CharSequence.h"
#include "CharClassifier.h"

struct AnnotatedLine : Doc
{
	AnnotatedLine()
		: Doc(),
		m_chClassifier(&CharClassifier::getInstance())
	{}

	AnnotatedLine(string pathLine)
		: Doc(pathLine),
		m_chClassifier(&CharClassifier::getInstance())
	{}

	struct scoresType
	{
		HogSvmModel m_hs_model;
		Mat m_scoreVals;
	};

	void InitCombinedImg(string pathImage, string binPath, Mat lineEnd, Mat lineEndBin);
	void computeFixedScores();
	void computeScores(const CharSequence *charSeq = nullptr, bool accTrans = true);
	void computeScore4Char(uchar asciiCode);

	unordered_map<uchar, scoresType> m_scores;
	Mat m_fixedScores;

	CharClassifier *m_chClassifier;
	CharSequence m_charSeq;
	string m_lineId;

	
};

#endif