#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

/************************************************************************
Project:  Chars Alignment
Module:   Classifier
Purpose:  Alignment discriminative algorithm

*************************** INCLUDE FILES ******************************/
#include "Dataset.h"
#include "CharClassifier.h"

class Classifier
{
public:
	Classifier(double _min_sqrt_gamma, std::string _loss_type, bool accTrans = true);
	void load(const std::string &filename);
	void save(const std::string &filename);
	bool was_changed() { return m_w_changed; }
	double update(AnnotatedLine& x, StartTimeSequence &y, StartTimeSequence &y_hat);
	double predict(AnnotatedLine& x, StartTimeSequence &y_hat, bool useCache = false);
	Mat phi(AnnotatedLine& x, StartTimeSequence& y);
	Mat phi_1(AnnotatedLine& x, int i, int t, int l);
	double phi_2(AnnotatedLine& x, int i, int t, int l1, int l2);
	static double gamma(const StartTimeSequence &y, const StartTimeSequence &y_hat);
	static double gamma(const int y, const int y_hat);
	static double gaussian(const double x, const double mean, const double std);
	double aligned_phoneme_scores(AnnotatedLine& x, StartTimeSequence &y);

public:
	static std::string m_loss_type;

protected:

	static int m_phi_size;
	Mat m_w;
	Mat m_w_old;
	bool m_w_changed;
	double m_min_sqrt_gamma;
	bool m_accTrans;

	TrainingData &m_trData;
};

#endif // _CLASSIFIER_H
