#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include "LMParams.h"

using namespace boost::filesystem;

void LMParams::initDirs()
{
	vector<const string*> dirs = { &m_pathData, &m_pathResultsParent, &m_pathResults, &m_pathResultsImages};

	for (auto d : dirs)
	{
		path p(*d);
		if (!exists(p))
		{
			boost::filesystem::create_directory(p);
		}
	}
}