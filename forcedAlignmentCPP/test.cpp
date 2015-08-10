#include <chrono>
#include <iostream>
#include "LearnModels.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	LearnModels lm;
	lm.LearnModelsAndEvaluate();

	cout << "Press any key to continue.";
	getchar();
	return 0;
}