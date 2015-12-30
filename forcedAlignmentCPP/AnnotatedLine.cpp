#include <set>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "AnnotatedLine.h"
#include "commonTypes.h"
#include "HogUtils.h"

using namespace boost::filesystem;
using namespace std;

void AnnotatedLine::InitCombinedImg(string pathImage, string binPath, Mat lineEnd, Mat lineEndBin)
{
	m_featuresComputed = false;
	Mat origImg = cv::imread(pathImage);

	m_pathImage = pathImage;
	if (!origImg.data)
	{
		std::cerr << "Could not open or find the image " << m_pathImage << std::endl;
	}

	m_origImage = Mat::zeros(std::max(origImg.rows, lineEnd.rows), origImg.cols + lineEnd.cols, CV_8UC3);
	lineEnd.copyTo(m_origImage(Range(0, lineEnd.rows), Range(0, lineEnd.cols)));
	origImg.copyTo(m_origImage(Range(0, origImg.rows), Range(lineEnd.cols, m_origImage.cols)));

	path p(m_pathImage);
	string binImgPath = binPath + p.stem().string() + ".png";
	Mat origBin = cv::imread(binImgPath, CV_LOAD_IMAGE_GRAYSCALE);

	m_bin = Mat::zeros(std::max(origImg.rows, lineEndBin.rows), origImg.cols + lineEndBin.cols, CV_8UC1);
	lineEndBin.copyTo(m_bin(Range(0, lineEndBin.rows), Range(0, lineEndBin.cols)));
	origBin.copyTo(m_bin(Range(0, origBin.rows), Range(lineEndBin.cols, m_bin.cols)));

	m_H = m_origImage.rows;
	m_W = m_origImage.cols;
}

void AnnotatedLine::computeFixedScores()
{
	// row 0 - projection profile (start char)
	// row 1 - connected components (start char)
	// row 2 - integral projection profile.
	// row 3 - projection profile (end char)
	// row 4 - connected components (end char)

	m_fixedScores = Mat::zeros(5, m_W, CV_64F);

	const int intervalSz = 5;

	// computing Projection Profile scores.

	Mat pp;
	reduce(m_bin, pp, 0, CV_REDUCE_SUM, CV_64F);
	Mat ppMat = m_fixedScores.rowRange(Range(0, 1));
	Mat ppEndMat = m_fixedScores.rowRange(Range(3, 4));
	for (int i = 0; i < pp.cols; ++i)
	{
		bool localMin = true;
		double currentVal = pp.at<double>(i);
		for (int offset = -intervalSz; (offset <= intervalSz) && localMin; ++offset)
		{
			double neighbourIdx = std::min(std::max(i + offset, 0), pp.cols);
			localMin &= currentVal <= pp.at<double>(neighbourIdx);
		}
		if (localMin && currentVal < pp.at<double>(std::min(i + 1, pp.cols)))
		{
			ppMat.colRange(std::max(i - intervalSz + 2, 0), std::min(int(i + intervalSz + 1), pp.cols)) = 1;
		}
		if (localMin && currentVal < pp.at<double>(std::max(i - 1, 0)))
		{
			ppEndMat.colRange(std::max(i - intervalSz + 2, 0), std::min(int(i + intervalSz + 1), pp.cols)) = 1;
		}
	}

	// computing score using CC's
	Mat ccMat = m_fixedScores.rowRange(Range(1, 2));
	Mat ccEndMat = m_fixedScores.rowRange(Range(4, 5));
	Mat labels;
	cv::connectedComponents(m_bin.t(), labels);
	transpose(labels, labels);

	set<int> current;
	set<int> next;
	set<int> diff;

	for (auto rowIdx = 0; rowIdx < labels.rows; ++rowIdx)
	{
		current.insert(labels.at<int>(rowIdx, 0));
	}

	for (auto colIdx = 1; colIdx < labels.cols - 1; ++colIdx)
	{
		for (auto rowIdx = 0; rowIdx < labels.rows; ++rowIdx)
		{
			next.insert(labels.at<int>(rowIdx, colIdx));
		}

		set_difference(next.begin(), next.end(), current.begin(), current.end(), inserter(diff, diff.begin()));
		if (!diff.empty())
		{
			ccMat.colRange(std::max(colIdx - intervalSz + 1, 0), std::min(int(colIdx + intervalSz), labels.cols)) = 1;
		}
		diff.clear();

		set_difference(current.begin(), current.end(), next.begin(), next.end(), inserter(diff, diff.begin()));
		if (!diff.empty())
		{
			ccEndMat.colRange(std::max(colIdx - intervalSz + 1, 0), std::min(int(colIdx + intervalSz), labels.cols)) = 1;
		}
		diff.clear();

		current = move(next);
		next.clear();
	}

	// computing integral Projection Profile
	Mat intPPMat;
	cv::integral(pp, intPPMat);
	intPPMat /= 255;
	intPPMat.rowRange(1, 2).colRange(1, pp.cols + 1).copyTo(m_fixedScores.rowRange(Range(2, 3)));
}

void AnnotatedLine::computeScores(const CharSequence *charSeq /*= nullptr */, bool accTrans /* = true */)
{
	// Is it the first time we compute scores ?
	if (nullptr == charSeq)
	{
		computeFeatures();
		computeFixedScores();
	}

	const CharSequence &cq = (nullptr == charSeq) ? m_charSeq : (*charSeq);

	for (auto asciiCode : cq)
	{
		set<uchar> asciiCodes = { asciiCode, accTrans ? asciiCode : (uchar)std::toupper(asciiCode) };

		for (auto asciiCode_ : asciiCodes)
		{
			computeScore4Char(asciiCode_);
		}
	}
}

void AnnotatedLine::computeScore4Char(uchar asciiCode)
{
			AnnotatedLine::scoresType scores;
			scores.m_scoreVals = Mat::ones(1, m_W, CV_64F)*-1;
	uint sbin = m_params->m_sbin;

	if (asciiCode != '|')
			{
				// Computing scores for this model over the line.
		if (m_scores.find(asciiCode) == m_scores.end())
				{
			scores.m_hs_model = m_chClassifier->learnModel(asciiCode);

					if (scores.m_hs_model.isInitialized())
					{
						vector<Rect> locW;
						vector<double> scsW;
						HogUtils::getWindows(*this, scores.m_hs_model, scsW, locW, m_params->m_step, sbin, false);

						// computing HOG scores.
						double *ptr = scores.m_scoreVals.ptr<double>(0);
						for (size_t i = 0; i < locW.size(); ++i)
						{
							int xInd = locW[i].x*sbin;
							ptr[xInd] = std::max(ptr[xInd], 10 * scsW[i]);
						}

						for (auto idx = 0; idx < scores.m_scoreVals.cols; ++idx)
						{
							ptr[idx] = ptr[idx - (idx % sbin)];
						}
					}
					else
					{
				clog << "No model for " << asciiCode << ", trying to continue." << endl;
					}
				}
		m_scores.insert({ asciiCode, move(scores) });
			}
		}
