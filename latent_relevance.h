// Copyright (c) 2014 Baidu Corporation
// @file:   latent_relevance.h
// @brief:  Header file of (factorized) latent relevance algorithm
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-07-17

namespace latent_relevance {

// Data struct
struct data_t {
	int y;						// Label
	float* x;					// x = [x1, x2], the size is 2 * vector_size;

	// Intermediate variables for speed-up
	float cos;					// Cosine(x1, x2)
	float score;				// Predicted score, P(y = 1 | w, x)
	float* x_ij;				// x1_i * x2_j
};

class LatentRelevance {
public:
	LatentRelevance();
	~LatentRelevance();

	// Set member variables
	void set_factorize_mode(int value);
	void set_k(int value);
	void set_c(float value);
	void set_learn_rate(float value);

	// Read data from input file
	int read_data(const char* file_name);

	// Initialize
	int initialize();

	// Pre-calculate intermediate variables for speed-up
	int pre_calculate();

	// Parse line to get data (x, y)
	int parse_line(char* buf, data_t* data);

	// Parse vector
	int parse_vector(char* buf, float* v);

	// Count the number of c in str
	int count_char_number(char* str, char c);

	// Normalize (L2 norm) input vector (in-place)
	int normalize_vector(float* x, int vector_size);

	// Train latent relevance model
	int train();

	// Run gradient descent for training
	int run_gd();

	// Calculate gradient
	float calculate_gradient_w0();
	float calculate_gradient_wcos();
	float calculate_gradient_w(int index1, int index2);
	float calculate_gradient_v(int index1, int index2, float* temp_gd);

	// Line search
	int line_search(float gradient_w0, float gradient_wcos, float* gradient_weights, float& loss);

	// Proximal mapping for L1 regularization
	float proximal_operator_L1(float weight);

	// Calculate current loss (negative log likelihood)
	float calculate_loss();

	// Predict input file
	int test(const char* file_name, const char* model_name);

	// Predict latent relevance
	float predict(data_t* data);

	// Save model
	int save_model(const char* model_name);

	// Load model for prediction
	int load_model(const char* model_name);

//private:
public:	// For debugging
	// Iteration stop condition
	static const int MAX_STOP_ITER_NUM;		// Max iteration number
	static const float MIN_STOP_LOSS;		// Stop when loss < MIN_STOP_LOSS
	static const float MAX_OPTIMIZE_STEP;	// Limit optimization step

	// Data
	int data_num;
	int vector_size;
	data_t* m_data;						// Training data

	// Factorization
	int factorize_mode;					// 1 - with factorization, 0 - no factorization
	int k;								// Factorization dimension
	
	// Model
	float* w;							// Model weights for factorize_mode = 0, the size is vector_size * vector_size
	float* v;							// Model weights (latent factors) for factorize_mode = 1, the size is 2 * vector_size * k
	float w0;							// Weight for bias(1)
	float w_cos;						// Weight for cosine(x1, x2)

	// Parameters
	float c;							// Regularization coefficient
	float learn_rate;					// Learning rate

	// Intermediate variables
	float pre_loss_step;				// Previous loss step: step_size * dir_grad
};

} // namespace latent_relevance

