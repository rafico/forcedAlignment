#include <numeric>
#include "TrainingChars.h"

TrainingChars::TrainingChars(const LMParams& params)
	: m_params(params)
{}

void TrainingChars::addCharInstance(const CharInstance& ci)
{
	auto& tch = m_charInstances[ci.m_asciiCode];
	tch.m_instances.push_back(ci);
}

const vector<CharInstance>& TrainingChars::getSamples(uint asciiCode)
{
	auto& tch = m_charInstances[asciiCode];
	return tch.m_instances;
}


void TrainingChars::computeNormalDistributionParams()
{
	// computing the mean.
	for (auto& ch : m_charInstances)
	{
		auto samplesVec = getSamples(ch.first);
		double widthSum = 0;
		double heightSum = 0;
		for_each(samplesVec.begin(), samplesVec.end(), [&](const CharInstance& ci)
		{
			widthSum += ci.m_loc.width;
			heightSum += ci.m_loc.height;
		});

		double meanWidth = (widthSum / samplesVec.size());
		double meanHeight = (heightSum / samplesVec.size());

		double accumWidth = 0.0;
		double accumHeight = 0.0;
		for_each(begin(samplesVec), end(samplesVec), [&](const CharInstance& ch)
		{
			accumWidth += (ch.m_loc.width - meanWidth) * (ch.m_loc.width - meanWidth);
			accumHeight += (ch.m_loc.height - meanHeight) * (ch.m_loc.height - meanHeight);
		});

		double varWidth = accumWidth / (samplesVec.size() - 1);
		double varHeight = accumHeight / (samplesVec.size() - 1);

		ch.second.m_widthMean = meanWidth + 2 * m_params.m_sbin;
		ch.second.m_widthVariance = varWidth;

		ch.second.m_heightMean = meanHeight + 2 * m_params.m_sbin;
		ch.second.m_heightVariance = varHeight;
	}
}

double TrainingChars::getMeanWidth(uint asciiCode)
{
	return m_charInstances[asciiCode].m_widthMean;
}

double TrainingChars::getMeanHeight(uint asciiCode)
{
	return m_charInstances[asciiCode].m_heightMean;
}