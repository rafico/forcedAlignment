#include <opencv2/highgui.hpp>
#include "Doc.h"
#include "HogUtils.h"

using namespace cv;

Doc::Doc()
{}

Doc::Doc(string pathImage, vector<CharInstance> && chars /* = vector<CharInstance>() */)
	: m_origImage(imread(pathImage)),
	m_pathImage(pathImage),
	m_chars(chars),
	m_nChars(m_chars.size()),
	m_H(m_origImage.rows),
	m_W(m_origImage.cols)
{
	if (!m_origImage.data)
	{
		std::cerr << "Could not open or find the image " << m_pathImage << std::endl;
	}
}

void Doc::Init(string pathImage)
{
	m_origImage = imread(pathImage);
	m_pathImage = pathImage;
	if (!m_origImage.data)
	{
		std::cerr << "Could not open or find the image " << m_pathImage << std::endl;
	}
	m_H = m_origImage.rows;
	m_W = m_origImage.cols;
}

void Doc::resizeDoc(uint sbin)
{
	uint H = m_H;
	uint W = m_W;
	uint res;

	while ((res = (H % sbin)))
	{
		H -= res;
	}
	while ((res = (W % sbin)))
	{
		W -= res;
	}
	uint difH = m_H - H;
	uint difW = m_W - W;
	uint padYini = difH / 2;
	uint padYend = difH - padYini;
	uint padXini = difW / 2;
	uint padXend = difW - padXini;

	const Mat &im = m_origImage;
	im(Range(padYini, im.rows - padYend), Range(padXini, im.cols - padXend)).copyTo(m_image);
	m_yIni = padYini;
	m_xIni = padXini;
	m_H = m_image.rows;
	m_W = m_image.cols;
}

void Doc::computeFeatures(uint sbin)
{
	Mat feat = HogUtils::process(m_image, sbin, &m_bH, &m_bW);
	feat.convertTo(m_features, CV_32F);
}
