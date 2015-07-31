#include <chrono>
#include <iostream>
#include "LearnModels.h"
#include "PedroFeatures.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	LearnModels lm;
	lm.loadModels();

	cout << "Press any key to continue.";
	auto c = getchar();
	return 0;
}