#include "ForcedAlignmentTrain.h"
#include "commonTypes.h"
#include "Dataset.h"
#include "Params.h"
#include "Classifier.h"
#include "CharClassifier.h"

void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat);

void ForcedAlignmentTrain::train()
{
	string classifier_filename = "classifier_weight.txt";

	double epsilon = 0;

	string loss_type = "alignment_loss";

	// Initiate classifier
	Classifier classifier(0, loss_type);

	uint num_epochs = 1;

	double loss;
	double cum_loss = 0.0;
	double best_validation_loss = 1e100;

	for (uint epoch = 0; epoch < num_epochs; ++epoch) {

		//beginning of the training set
		Dataset dataset;

		double max_loss_in_epoch = 0.0; // maximal loss value in this epoch
		double avg_loss_in_epoch = 0.0; // training loss value in this epoch

		// Run over all dataset
		for (uint i = 0; i < dataset.trSize(); i++)
		{
			AnnotatedLine x;
			StartTimeSequence y;
			StartTimeSequence y_hat;

			cout << "==================================================================================" << endl;

			// read next example for dataset
			dataset.readTrLine(x, y);
			
			y_hat.resize(y.size());

			// predict label
			classifier.predict(x, y_hat);

			cout << "chars=" << x.m_charSeq << endl;
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
				dataset.resetValIndex();
				cout << "Validation...\n";
				double this_w_loss = 0.0;
				for (uint ii = 0; ii < dataset.valSize(); ++ii) 
				{
					AnnotatedLine xx;
					StartTimeSequence yy;
					StartTimeSequence yy_hat;
					dataset.readValLine(xx, yy);
					yy_hat.resize(yy.size());
					classifier.predict(xx, yy_hat);
					//drawSegResult(xx, yy, yy_hat);
					double this_loss = classifier.gamma(yy, yy_hat);
					this_w_loss += this_loss;
				}
				this_w_loss /= dataset.valSize();
				if (this_w_loss < best_validation_loss) 
				{
					best_validation_loss = this_w_loss;
					classifier.save(classifier_filename);
				}
				cout << "i = " << i << ", this validation error = " << this_w_loss
					<< ", best validation loss  = " << best_validation_loss << endl;

				// stopping criterion for iterate until convergence
				//        if (best_validation_loss < 1.0)
				//          break;
			}

		} // end running over the dataset

		avg_loss_in_epoch /= dataset.valSize();

		cout << " average normalized loss = " << avg_loss_in_epoch
			<< " best validation loss  = " << best_validation_loss << endl;
	}

	cout << "Done." << endl;
}

void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat)
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
