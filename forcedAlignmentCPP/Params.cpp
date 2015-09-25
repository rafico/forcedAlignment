#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include "Params.h"

using namespace boost::filesystem;

void Params::initDirs()
{
	vector<const string*> dirs = { &m_pathData, &m_pathResultsParent, &m_pathResults, &m_pathResultsImages, &m_pathCharModels };

	for (auto d : dirs)
	{
		path p(*d);
		if (!exists(p))
		{
			boost::filesystem::create_directory(p);
		}
	}
}