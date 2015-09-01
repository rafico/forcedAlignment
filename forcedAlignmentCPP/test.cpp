#include <chrono>
#include <iostream>
#include "LearnModels.h"
#include "JonAlmazan.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	LearnModels lm;
	lm.learnModels();
	lm.evaluateModels();

	//JonAlmazan ja;
	//ja.LearnModelsAndEvaluate();

	cout << "Press any key to continue.";
	getchar();
	return 0;
}