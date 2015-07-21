#include <chrono>
#include <iostream>
#include "LearnModels.h"
#include "PedroFeatures.h"

using namespace std;

int main(int argc, char** argv)
{
	/* testLineExtraction(); */
	//TODO: clean up the dataset of character, it contains noise.

	LearnModels lm;
	lm.train();


	cout << "Press any key to continue.";
	auto c = getchar();
	return 0;
}