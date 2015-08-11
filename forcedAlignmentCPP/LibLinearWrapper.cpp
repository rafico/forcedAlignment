#include "LibLinearWrapper.h"


LibLinearWrapper::LibLinearWrapper()
{
	m_param.solver_type = L2R_L2LOSS_SVC;
	//m_param.solver_type = L2R_L1LOSS_SVC_DUAL;
	m_param.eps = 0.01;
	m_param.C = 1;
	m_param.nr_weight = 0;
	m_param.weight_label = nullptr;
	m_param.weight = nullptr;
	m_param.p = 0.1;
	m_param.init_sol = nullptr;

}

void LibLinearWrapper::trainModel(Mat labels, Mat trainingData, vector<float>& weight)
{
	CV_Assert(labels.rows == trainingData.rows);
	m_prob.l = labels.rows;
	m_prob.n = trainingData.cols;

	auto num_samples = trainingData.total();
	auto elements = num_samples + m_prob.l * 2;
	
	vector<double> y(m_prob.l);
	vector<struct feature_node*> x(m_prob.l);
	vector<struct feature_node> x_spaceVec (elements);

	m_prob.y = y.data();
	m_prob.x = x.data();
	auto x_space = x_spaceVec.data();

	m_prob.bias = -1;

	int j = 0;
	for (int i = 0; i<m_prob.l; ++i)
	{
		m_prob.x[i] = &x_space[j];
		m_prob.y[i] = labels.at<float>(i);

		auto row_ptr = trainingData.ptr<float>(i);
		for (int k = 0; k < trainingData.cols; ++k)
		{
			if (row_ptr[k])
			{
				x_space[j].index = k + 1;
				x_space[j].value = row_ptr[k];
				++j;
			}
		}
		x_space[j++].index = -1;
	}

	auto error_msg = check_parameter(&m_prob, &m_param);
	if (error_msg)
	{
		std::clog << "Error: " << error_msg << std::endl;
		return;
	}
	struct model *model_ = train(&m_prob, &m_param);
	weight.resize(model_->nr_feature);
	for (size_t i = 0; i < weight.size(); ++i)
	{
		weight[i] = static_cast<float>(model_->w[i]);
	}
}