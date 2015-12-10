#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "HogUtils.h"

#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
double uu[9] = { 1.0000,
0.9397,
0.7660,
0.500,
0.1736,
-0.1736,
-0.5000,
-0.7660,
-0.9397 };
double vv[9] = { 0.0000,
0.3420,
0.6428,
0.8660,
0.9848,
0.9848,
0.8660,
0.6428,
0.3420 };


using namespace std;
using namespace cv;

// main function:
// takes a double color image and a bin size 
// returns HOG features
// image should be color
// based on Pedro Felzenszwalb's work -  http://www.cs.berkeley.edu/~rbg/latent/index.html

cv::Mat HogUtils::process(cv::Mat image, int sbin, int *h /*=0*/, int *w /*=0*/)
{
	CV_Assert(image.channels() == 3 && image.isContinuous());
	if (image.type() != CV_64F)
	{
		image.convertTo(image, CV_64F);
	}
	int dims[] = { image.rows, image.cols };
	double *im = image.ptr<double>(0);

	// memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = (int)round((double)dims[0] / (double)sbin);
	blocks[1] = (int)round((double)dims[1] / (double)sbin);
	auto histVec = vector<double>(blocks[0] * blocks[1] * 18);
	auto normVec = vector<double>(blocks[0] * blocks[1]);
	double *hist = histVec.data();
	double *norm = normVec.data();

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0] - 2, 0);
	out[1] = max(blocks[1] - 2, 0);
	out[2] = 27 + 4;
	Mat feat(out[0] * out[1], out[2], CV_64F);

	int visible[2];
	visible[0] = blocks[0] * sbin;
	visible[1] = blocks[1] * sbin;

	for (int y = 1; y < visible[0] - 1; y++) {
		for (int x = 1; x < visible[1] - 1; x++) {

			// first color channel
			double *s = im + min(y, dims[0] - 2)*dims[1] * 3 + min(x, dims[1] - 2) * 3;
			double dx3 = *(s + 3) - *(s - 3);
			double dy3 = *(s + 3 * dims[1]) - *(s - 3 * dims[1]);
			double v3 = dx3*dx3 + dy3*dy3;

			// second color channel
			++s;
			double dx2 = *(s + 3) - *(s - 3);
			double dy2 = *(s + 3 * dims[1]) - *(s - 3 * dims[1]);
			double v2 = dx2*dx2 + dy2*dy2;

			// third color channel
			++s;
			double dx = *(s + 3) - *(s - 3);
			double dy = *(s + 3 * dims[1]) - *(s - 3 * dims[1]);
			double v = dx*dx + dy*dy;

			// pick channel with strongest gradient
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			// snap to one of 18 orientations
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++) {
				double dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o + 9;
				}
			}

			// add to 4 histograms around pixel using linear interpolation
			double xp = ((double)x + 0.5) / (double)sbin - 0.5;
			double yp = ((double)y + 0.5) / (double)sbin - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			double vx0 = xp - ixp;
			double vy0 = yp - iyp;
			double vx1 = 1.0 - vx0;
			double vy1 = 1.0 - vy0;
			v = sqrt(v);

			if (ixp >= 0 && iyp >= 0) {
				*(hist + ixp*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx1*vy1*v;
			}

			if (ixp + 1 < blocks[1] && iyp >= 0) {
				*(hist + (ixp + 1)*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx0*vy1*v;
			}

			if (ixp >= 0 && iyp + 1 < blocks[0]) {
				*(hist + ixp*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx1*vy0*v;
			}

			if (ixp + 1 < blocks[1] && iyp + 1 < blocks[0]) {
				*(hist + (ixp + 1)*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx0*vy0*v;
			}
		}
	}

	// compute energy in each block by summing over orientations
	for (int o = 0; o < 9; o++) {
		double *src1 = hist + o*blocks[0] * blocks[1];
		double *src2 = hist + (o + 9)*blocks[0] * blocks[1];
		double *dst = norm;
		double *end = norm + blocks[1] * blocks[0];
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	double *pFeat = feat.ptr<double>(0);

	// compute features
	for (int x = 0; x < out[1]; x++) {
		for (int y = 0; y < out[0]; y++) {
			double *dst = pFeat + (y*out[1] + x)*out[2];
			double *src, *p, n1, n2, n3, n4;

			p = norm + (x + 1)*blocks[0] + y + 1;
			n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + (x + 1)*blocks[0] + y;
			n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y + 1;
			n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y;
			n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 18; o++) {
				double h1 = min(*src * n1, 0.2);
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst++ = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				src += blocks[0] * blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 9; o++) {
				double sum = *src + *(src + 9 * blocks[0] * blocks[1]);
				double h1 = min(sum * n1, 0.2);
				double h2 = min(sum * n2, 0.2);
				double h3 = min(sum * n3, 0.2);
				double h4 = min(sum * n4, 0.2);
				*dst++ = 0.5 * (h1 + h2 + h3 + h4);
				src += blocks[0] * blocks[1];
			}

			// texture features
			*dst++ = 0.2357 * t1;
			*dst++ = 0.2357 * t2;
			*dst++ = 0.2357 * t3;
			*dst++ = 0.2357 * t4;
		}
	}

	if (h && w)
	{
		*h = out[0];
		*w = out[1];
	}

	return feat;
}

// Extracts all the possible patches from a document images and computes the
// score given the queried word model
void HogUtils::getWindows(const Doc& doc, const HogSvmModel& hs_model, vector<double>& scsW, vector<Rect>& locW, uint step, uint sbin, bool padResult /* = true */)
{
	//TODO: optimize this function (try using GPU).

	uint xIni = doc.m_xIni, yIni = doc.m_yIni;
	uint widthPad = 2, heightPad = 2;
	uint p_sbin = sbin;
	if (!padResult)
	{
		xIni = yIni = widthPad = heightPad = 0;
		p_sbin = 1;
	}

	int bH, bW;
	Mat featDoc;
	const_cast<Doc &>(doc).getComputedFeatures(featDoc, bH, bW, sbin);

	CV_Assert(featDoc.isContinuous());
	float *flat = featDoc.ptr<float>(0);
	auto nbinsH = hs_model.m_bH;
	auto nbinsW = hs_model.m_bW;
	auto dim = hs_model.weight.size() / (nbinsH*nbinsW);
	auto numWindows = (bH - nbinsH + 1) * (bW - nbinsW + 1);
	scsW.resize(numWindows);

	size_t k = 0;
	for (auto by = 0; by <= bH - nbinsH; by += step)
	{
		for (auto bx = 0; bx <= bW - nbinsW; bx += step)
		{
			scsW[k] = 0;
			double norm = 0;

			for (auto tmpby = by, i = 0; tmpby < by + nbinsH; ++tmpby, ++i)
			{
				auto pos = (tmpby*bW + bx)*dim;
				scsW[k] += inner_product(hs_model.weight.data() + i*nbinsW*dim, hs_model.weight.data() + (i + 1)*nbinsW*dim, flat + pos, .0);
				norm += inner_product(flat + pos, flat + pos + nbinsW*dim, flat + pos, .0);
			}
			scsW[k++] /= sqrt(norm);
			locW.push_back(Rect(bx*p_sbin + xIni, by*p_sbin + yIni, (nbinsW + widthPad)*p_sbin, (nbinsH + heightPad)*p_sbin));
		}
	}

	for_each(scsW.begin(), scsW.end(), [](double &scs){if (std::isnan(scs)) scs = -1; });
}

vector<int> HogUtils::nms(Mat I, const vector<Rect>& X, double overlap)
{
	int i, j;
	int N = I.rows;

	vector<int> used(N, 0);
	vector<int> pick(N, -1);

	int Npick = 0;
	for (i = N - 1; i >= 0; i--)
	{
		if (!used[i])
		{
			used[i] = 1;
			pick[Npick++] = I.at<int>(i);
			for (j = 0; j < i; ++j)
			{
				const Rect& r_i = X[I.at<int>(i)];
				const Rect& r_j = X[I.at<int>(j)];
				if (!used[j] && (double((r_i & r_j).area()) / r_j.area()) >= overlap)
				{
					used[j] = 1;
				}
			}
		}
	}
	pick.resize(Npick);
	return pick;
}
void HogUtils::getFeaturesStartingFromColumn(const Doc& doc, uint& col, uint sbin, Mat& features)
{
	int bH, bW;
	Mat featDoc;
	const_cast<Doc &>(doc).getComputedFeatures(featDoc, bH, bW, sbin);
	col -= col%sbin;

	uint start_col = col / sbin;
	int numCols = bW - start_col;
	features.create(numCols*bH, 31, CV_32F);
	
	for (auto by = 0; by < bH; ++by)
	{
		auto a = by*bW + start_col;
		auto b = (by + 1)*bW;
		auto c = by*numCols;
		auto d = (by + 1)*numCols;
		featDoc.rowRange(Range(a, b)).copyTo(features.rowRange(Range(c, d)));
	}
}
