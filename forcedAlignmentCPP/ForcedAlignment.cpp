#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include "ForcedAlignment.h"
#include "commonTypes.h"
#include "Params.h"
#include "Classifier.h"
#include "CharClassifier.h"
#include "HogUtils.h"

using namespace boost::filesystem;

ForcedAlignment::ForcedAlignment()
	: m_classifier_filename("classifier_weight.txt"),
	m_params(Params::getInstance())
{
	path p(m_params.m_pathResultsImages + "alignment/");
	if (!exists(p.parent_path()))
	{
		boost::filesystem::create_directory(p.parent_path());
	}

	m_resultFile.open(p.string() + "result.txt", std::fstream::app);
	m_resultFischerFile.open(p.string() + "resultFischerFile.txt", std::fstream::out);
}

ForcedAlignment::~ForcedAlignment()
{
	m_resultFile.close();
	m_resultFischerFile.close();
}

void ForcedAlignment::train()
{
	//double epsilon = 0;

	string loss_type = "alignment_loss";
	string training_file_list = "train.txt";
	string validation_file_list = "valid.txt";
	string start_times_file = "charStartTime.txt";

	// Initiate classifier
	Classifier classifier(0, loss_type);

	uint num_epochs = 1;

	double loss;
	double cum_loss = 0.0;
	double best_validation_loss = 1e100;

	for (uint epoch = 0; epoch < num_epochs; ++epoch) {

		//beginning of the training set
		Dataset training_dataset(training_file_list, start_times_file);

		double max_loss_in_epoch = 0.0; // maximal loss value in this epoch
		double avg_loss_in_epoch = 0.0; // training loss value in this epoch

		// Run over all dataset
		for (uint i = 0; i < training_dataset.size(); i++)
		{
			AnnotatedLine x;
			StartTimeSequence y;
			StartTimeSequence y_hat;

			cout << "==================================================================================" << endl;

			// read next example for dataset
			training_dataset.read(x, y);
			
			y_hat.resize(y.size());

			// predict label
			classifier.predict(x, y_hat);

			cout << "alignment= " << y << endl;
			cout << "predicted= " << y_hat << endl;

			//drawSegResult(x, y, y_hat);

			loss = classifier.update(x, y, y_hat);
			cum_loss += loss;

			if (max_loss_in_epoch < loss) max_loss_in_epoch = loss;
			avg_loss_in_epoch += loss;

			// now, check the validations error
			if (classifier.was_changed()) 
			{
				cout << "Validation...\n";
				Dataset val_dataset(validation_file_list, start_times_file);
				double this_w_loss = 0.0;
				for (uint ii = 0; ii < val_dataset.size(); ++ii)
				{
					AnnotatedLine xx;
					StartTimeSequence yy;
					StartTimeSequence yy_hat;
					val_dataset.read(xx, yy);
					yy_hat.resize(yy.size());
					classifier.predict(xx, yy_hat);
					//drawSegResult(xx, yy, yy_hat);
					double this_loss = classifier.gamma(yy, yy_hat);
					this_w_loss += this_loss;
				}
				this_w_loss /= val_dataset.size();
				if (this_w_loss < best_validation_loss) 
				{
					best_validation_loss = this_w_loss;
					classifier.save(m_classifier_filename);
				}
				cout << "i = " << i << ", this validation error = " << this_w_loss
					<< ", best validation loss  = " << best_validation_loss << endl;

				// stopping criterion for iterate until convergence
				//        if (best_validation_loss < 1.0)
				//          break;
			}

		} // end running over the dataset

		avg_loss_in_epoch /= training_dataset.size();

		cout << " average normalized loss = " << avg_loss_in_epoch
			<< " best validation loss  = " << best_validation_loss << endl;
	}

	cout << "Done." << endl;
}

void ForcedAlignment::decode()
{
	string test_file_list = "test.txt";
	string loss_type = "alignment_loss";
	
	// Initiate classifier

	Classifier classifier(0, loss_type);
	classifier.load(m_classifier_filename);
	
	// beginning of the test set
	Dataset test_dataset(test_file_list, "");

	// Run over all dataset
	for (uint i = 0; i < test_dataset.size(); i++) 
	{
		AnnotatedLine x;
		StartTimeSequence y;
		StartTimeSequence y_hat;

		cout << "==================================================================================" << endl;

		// read next example for dataset
		test_dataset.read(x, y, i);
		y_hat.resize(x.m_charSeq.size());

		// predict label 
		double confidence = classifier.predict(x, y_hat);
		if (test_dataset.labels_given())
		{
			cout << "alignment= " << y << endl;
		}
		// cout << "confidence= " << confidence << endl;
		drawSegResult(x, y_hat, m_params.m_pathResultsImages+"alignment/"+ x.m_lineId + ".png");
		m_resultFile << x.m_lineId << " " << confidence << " " << y_hat << endl;
		printConfidencePerChar(x, y_hat);
		
		m_resultFischerFile << x.m_lineId << " ";
		bool startWord = true;
		for (size_t i = 1; i < y_hat.size(); ++i)
		{
			if (x.m_charSeq[i] == '|' || startWord)
			{
				char delim = startWord ? '-' : '|';
				m_resultFischerFile << y_hat[i] << delim;
				startWord = x.m_charSeq[i] == '|';
			}
		}
		m_resultFischerFile << endl;

		//cout << "aligned_phoneme_score= " << classifier.aligned_phoneme_scores(x, y_hat) << endl;
#if 0    
		int end_frame = 0;
		confidence = classifier.align_keyword(x, y_hat, end_frame);
		cout << "k_predicted= " << y_hat << " " << end_frame << endl;
		cout << "k_confidence= " << confidence << endl;
#endif    


		/*
		// calculate the error
		if (test_dataset.labels_given()) 
		{
			int file_loss = 0;
			int cur_loss;
			for (unsigned int j = 0; j < y.size(); ++j) {
				if (y[j] > y_hat[j]) {
					cur_loss = y[j] - y_hat[j];
				}
				else {
					cur_loss = y_hat[j] - y[j];
				}
				file_loss += cur_loss;
				cummulative_loss += cur_loss;
				for (int t = 1; t <= NUM_CUM_LOSS_RESOLUTIONS; t++)
					if (cur_loss <= t) cum_loss_less_than[t]++;
			}
			num_boundaries += y.size();
			cout << "File loss = " << file_loss / double(y.size()) << endl;
			cout << "Cum loss = " << cummulative_loss / double(num_boundaries) << endl;
			for (uint t = NUM_CUM_LOSS_RESOLUTIONS; t >= 1; t--) {
				cout << "% Boundaries (t <= " << t*frame_rate << "ms) = "
					<< 100.0*cum_loss_less_than[t] / double(num_boundaries) << "\n";
			}
			cout << endl;
			Classifier::loss_type = "tau_insensitive_loss";
			cummulative_loss_eps_insensitive += (Classifier::gamma(y, y_hat)*y.size());
			//      cout << "Classifier::gamma(y, y_hat)=" << Classifier::gamma(y, y_hat) << " " << num_boundaries << endl;
			cout << classifier.loss_type << "=" << cummulative_loss_eps_insensitive / double(num_boundaries) << endl;
			Classifier::loss_type = "alignment_loss";
			cummulative_loss_eps_alignment += (Classifier::gamma(y, y_hat)*y.size());
			//      cout << "Classifier::gamma(y, y_hat)=" << Classifier::gamma(y, y_hat) << " " << num_boundaries << endl;
			cout << classifier.loss_type << "=" << cummulative_loss_eps_alignment / double(num_boundaries) << endl;

		}
		*/
	}
	/*
	if (output_confidence != "" && output_confidence_ofs.good())
		output_confidence_ofs.close();
	*/

	cout << "Done." << endl;
}

void ForcedAlignment::inAccDecode(const TranscriptLexicon& tl)
{
	const Params &params = Params::getInstance();

	string test_file_list = "test.txt";
	string loss_type = "alignment_loss";
	std::ofstream m_resultFile;
	std::ofstream m_resultFischerFile;

	path p(params.m_pathResultsImages + "alignment/");
	if (!exists(p.parent_path()))
	{
		boost::filesystem::create_directory(p.parent_path());
	}

	// Initiate classifier

	Classifier classifier(0, loss_type, false);
	classifier.load(m_classifier_filename);

	// beginning of the test set
	Dataset test_dataset(test_file_list, "", false);

	Mat lineEnd;
	Mat lineEndBin;
	AnnotatedLine prevX;

	// Run over all dataset
	for (uint i = 0; i < test_dataset.size(); ++i)
	{
		AnnotatedLine x;
		StartTimeSequence y;
		StartTimeSequence y_hat;

		cout << "==================================================================================" << endl;

		// read next example for dataset
		test_dataset.read(x, lineEnd, lineEndBin, y, i);
		
		double confidence;
		vector<CharSequence> variations = tl.getPossibleLineVariations(x.m_charSeq);
		size_t maxIdx = 0;
		double maxConfidence=0;
		StartTimeSequence y_max;
		for (size_t cs_idx = 0; cs_idx < variations.size(); ++cs_idx)
		{
			auto &cs = variations[cs_idx];
			x.m_charSeq = cs;
		// predict label 
		y_hat.resize(x.m_charSeq.size());
			confidence = classifier.predict(x, y_hat);
			cout << "confidence: " << confidence << endl;
			if (confidence > maxConfidence)
			{
				maxConfidence = confidence;
				maxIdx = cs_idx;
				y_max = y_hat;
			}
		}

		cout << "max confidence: " << maxIdx << endl;
		x.m_charSeq = variations[maxIdx];
		y_hat = y_max;

		if (test_dataset.labels_given())
		{
			cout << "alignment= " << y << endl;
		}
		// cout << "confidence= " << confidence << endl;
		drawSegResult(x, y_hat, params.m_pathResultsImages + "alignment/" + x.m_lineId + ".png");
		//m_resultFile << x.m_lineId << " " << confidence << " " << y_hat << endl;
		//printConfidencePerChar(x, y_hat);

		// Have we located chars in the previous lineEnd ?
		// $last_loc_from_prev_line$ indicate the last entry in the alignment vector, y, which is located in the previous line part. 
		uint last_loc_from_prev_line;
		for (last_loc_from_prev_line = 0; y_hat[last_loc_from_prev_line + 1] < lineEnd.cols; ++last_loc_from_prev_line);
		cout << "last_loc_from_prev_line " << last_loc_from_prev_line << endl;

		if (last_loc_from_prev_line)
		{
			// recomputing the alignmnet for the previous line.
			//TODO: optimize by using previously computed values.
			cout << "Recomputing the alignmnet for the previous line with added chars\n" << endl;
			CharSequence addedChars;
			addedChars.assign(x.m_charSeq.begin() + 1, x.m_charSeq.begin() + last_loc_from_prev_line + 1);
			prevX.computeScores(&addedChars, false);
			prevX.m_charSeq.insert(prevX.m_charSeq.end(), addedChars.begin(), addedChars.end());
			prevX.m_charSeq.push_back('|');
			StartTimeSequence prev_y_hat;
			prev_y_hat.resize(prevX.m_charSeq.size());
			confidence = classifier.predict(prevX, prev_y_hat, true);
			drawSegResult(prevX, prev_y_hat, params.m_pathResultsImages + "alignment/" + prevX.m_lineId + ".png");
		}
			
			// recomputing the alignment for the current line.
		cout << "Recomputing alignment for line " << i << endl;
			test_dataset.read(x, y, i);
		x.m_charSeq = variations[maxIdx];
		x.m_charSeq.erase(x.m_charSeq.begin() + 1, x.m_charSeq.begin() + last_loc_from_prev_line + 1);
		//cout << "before: " << y_hat << endl;
		//cout << "lineEnd.cols: " << lineEnd.cols << endl;
		y_hat.erase(y_hat.begin() + 1, y_hat.begin() + last_loc_from_prev_line + 1);
		transform(y_hat.begin(), y_hat.end(), y_hat.begin(), [&](int stime){return std::max(stime - lineEnd.cols, 0); });
		//cout << "after: " << y_hat << endl;
			drawSegResult(x, y_hat, params.m_pathResultsImages + "alignment/" + x.m_lineId + ".png");

		auto lastCol = x.getRightmostBinCol();
		int lineEndCol = std::min(lastCol + 2, x.m_image.cols);

		if (lineEndCol > y_hat.back())
		{
			lineEnd = x.m_image.colRange(Range(y_hat.back(), lineEndCol));
			lineEndBin = x.m_bin.colRange(Range(y_hat.back(), lineEndCol));
		}
		else
		{
			lineEndBin = lineEnd = Mat();
		}
		prevX = x;
		//cout << "aligned_phoneme_score= " << classifier.aligned_phoneme_scores(x, y_hat) << endl;
	}

	m_resultFischerFile << "*------------------------ Done ----------------------------------*" << endl;
	for (auto &line : m_resultFischerCache)
	{
		m_resultFischerFile << line.first << " " << line.second << endl;
	}

	cout << "Done." << endl;
}

void ForcedAlignment::drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat)
{
	Mat img = x.m_image.clone();
	for (size_t i = 0; i < y.size(); ++i)
	{
		if (y[i] == y_hat[i])
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), cv::Scalar(255, 255, 0));
		}
		else
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), cv::Scalar(255, 0, 0));
			line(img, cv::Point(y_hat[i], 0), cv::Point(y_hat[i], img.rows), cv::Scalar(0, 255, 0));
		}
	}
}

void ForcedAlignment::drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, string resultPath)
{
	cv::Scalar green(0, 255, 0);
	cv::Scalar red(0, 0, 255);
	//cv::Scalar blue(255, 0, 0);
	Mat img = x.m_image.clone();
	bool startWord = true;
	int lineWidth = 1;

	for (size_t i = 1; i < y.size(); ++i)
	{
		if (x.m_charSeq[i] == '|' || startWord)
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), red, lineWidth);
			startWord = x.m_charSeq[i] == '|';
		}
		else
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), green, lineWidth);
		}
	}

	path p(resultPath);
	if (!exists(p.parent_path()))
	{
		boost::filesystem::create_directory(p.parent_path());
	}
	imwrite(resultPath, img);

	std::stringstream tempSstream;
	startWord = true;
	for (size_t i = 1; i < y.size(); ++i)
	{
		if (x.m_charSeq[i] == '|' || startWord)
		{
			char delim = startWord ? '-' : '|';
			tempSstream << y[i] << delim;
			startWord = x.m_charSeq[i] == '|';
		}
	}
	m_resultFischerCache.insert({ x.m_lineId, tempSstream.str() });

	m_resultFischerFile << x.m_lineId << " " << tempSstream.str() << endl;
}

void ForcedAlignment::writeResultInFischerFormat(std::ofstream& m_resultFischerFile, const AnnotatedLine &x, StartTimeSequence &y_hat)
{
	m_resultFischerFile << x.m_lineId << " ";
	bool startWord = true;
	for (size_t i = 1; i < y_hat.size(); ++i)
	{
		if (x.m_charSeq[i] == '|' || startWord)
		{
			char delim = startWord ? '-' : '|';
			m_resultFischerFile << y_hat[i] << delim;
			startWord = x.m_charSeq[i] == '|';
		}
	}
	m_resultFischerFile << endl;
}

void ForcedAlignment::printConfidencePerChar(AnnotatedLine &x, const StartTimeSequence &y)
{
	for (size_t i = 1; i < y.size(); ++i)
	{
		uchar asciiCode = x.m_charSeq[i];
		if (asciiCode != '|')
		{
			const auto& scores = x.m_scores[asciiCode];
			if (asciiCode != '|')
			{
				cout << x.m_charSeq[i] << ": " << scores.m_scoreVals.at<double>(0, y[i]) << endl;
			}
		}
	}
}
