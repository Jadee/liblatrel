// Copyright (c) 2014 Baidu Corporation
// @file:   latent_relevance.cpp
// @brief:  Source file of (factorized) latent relevance algorithm
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-07-17

#include "latent_relevance.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

namespace latent_relevance {

// Static member variables initialize
const int LatentRelevance::MAX_STOP_ITER_NUM = 20;
const float LatentRelevance::MIN_STOP_LOSS = 1e-3;

LatentRelevance::LatentRelevance() : data_num(0), vector_size(0), m_data(NULL), factorize_mode(0), k(0), 
									w(NULL), w0(0.0f), w_cos(0.0f), v(NULL), c(0.0f), learn_rate(0.0f),
									pre_loss_step(0.0f)
{
}

LatentRelevance::~LatentRelevance()
{
	// Free data
	for (int i = 0; i < data_num; ++i) {
		delete m_data[i].x;
		m_data[i].x = NULL;
		if (factorize_mode == 0) {
			delete m_data[i].x_ij;
			m_data[i].x_ij = NULL;
		}
	}
	delete m_data;
	m_data = NULL;

	// Free weights
	if (factorize_mode == 0) {
		delete w;
		w = NULL;
	}
	else {
		delete v;
		v = NULL;
	}
}

void LatentRelevance::set_factorize_mode(int value)
{
	factorize_mode = value;
}

void LatentRelevance::set_k(int value)
{
	k = value;
}

void LatentRelevance::set_c(float value)
{
	c = value;
}

void LatentRelevance::set_learn_rate(float value)
{
	learn_rate = value;
}

int LatentRelevance::read_data(const char* file_name)
{
	// Input file format: label \t x1 \t x2 (example: 1 \t 0.1 0.25 \t 0.3 0.1)
	FILE* fp = fopen(file_name, "r");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Reading data failed!\n", file_name);
		return -1;
	}

	const int MAX_LINE_DATA_LEN = 102400;
	char buf[MAX_LINE_DATA_LEN];

	// Count data number, and allocate data memory
	data_num = 0;
	while (fgets(buf, MAX_LINE_DATA_LEN, fp) != NULL) {
		++data_num;
	}
	
	// Initialize data
	m_data = new data_t[data_num];
	for (int i = 0; i < data_num; ++i) {
		m_data[i].y = 0;
		m_data[i].x = NULL;
		m_data[i].cos = 0.0f;
		m_data[i].score = 0.0f;
		m_data[i].x_ij = NULL;
	}

	rewind(fp);							// Back to the head of the file

	// Parse line data
	int line_num = 0;
	data_num = 0;						// Reset data number (drop invalid line data)
	while (fgets(buf, MAX_LINE_DATA_LEN, fp) != NULL) {
		++line_num;
		if (parse_line(buf, m_data + data_num) != 0) {
			printf("[WARNING] Parsing line %d failed!\n", line_num);
			continue;
		}

		++data_num;
	}

	// Initialize w and v
	if (factorize_mode == 0) {
		if (w == NULL) {
			w = new float[vector_size*vector_size];
			for (int i = 0; i < vector_size * vector_size; ++i) {
				w[i] = 0.0f;
			}
		}
	}
	else if (factorize_mode == 1) {
		if (v == NULL) {
			v = new float[2*vector_size*k];
			for (int i = 0; i < 2 * vector_size * k; ++i) {
				v[i] = 0.0f;
			}
		}
	}

	fclose(fp);
	return 0;
}

int LatentRelevance::parse_line(char* buf, data_t* data)
{
	// Parse label
	char* ptr = strtok(buf, "\t");
	if(ptr == NULL) {   
		printf("[WARNING] Parsing label failed!\n");
		return -1;
	} 
	data->y = static_cast<int>(strtol(ptr, NULL, 10));
	if (data->y != 1 && data->y != -1 && data->y != 0) {
		printf("[WARNING] Invalid label (should be 1(positive) or -1/0(negative))!\n");
		return -1;
	}

	// Parse x1
	char* x1 = strtok(NULL, "\t");
	if (x1 == NULL) {
		printf("[WARNING] Parsing x1 failed!\n");
		return -1;
	}

	// Parse x2
	char* x2 = strtok(NULL, "\t");
	if (x2 == NULL) {
		printf("[WARNING] Parsing x2 failed!\n");
		return -1;
	}

	// Get vector_size when parsing the 1st line
	if (vector_size == 0) {
		int x1_size = count_char_number(x1, ' ');
		int x2_size = count_char_number(x2, ' ');
		if (x1_size != x2_size or x1_size < 1) {
			printf("[WARNING] Invalid vector x1 and x2!\n");
			return -1;
		}
		else {
			vector_size = x1_size + 1;
		}
	}
	else {
		int x1_size = count_char_number(x1, ' ');
		if (vector_size != x1_size + 1) {
			printf("[WARNING] Wrong data vector size!");
			return -1;
		}
	}

	// Allocate data memory
	data->x = new float[2 * vector_size];
	if (factorize_mode == 0) {
		data->x_ij = new float[vector_size * vector_size];
	}

	// Parse x1 elements
	if (parse_vector(x1, data->x) != vector_size) {
		printf("[WARNING] Parsing x1 failed!\n");
		delete data->x;
		data->x = NULL;
		if (factorize_mode == 0 && data->x_ij != NULL) {
			delete data->x_ij;
			data->x_ij = NULL;
		}
		return -1;
	}

	// Parse x2 elements
	if (parse_vector(x2, data->x + vector_size) != vector_size) {
		printf("[WARNING] Parsing x2 failed!\n");
		delete data->x;
		data->x = NULL;
		if (factorize_mode == 0 && data->x_ij != NULL) {
			delete data->x_ij;
			data->x_ij = NULL;
		}
		return -1;
	}

	return 0;
}

int LatentRelevance::count_char_number(char* str, char c)
{
	int num = 0;
	while (*str != '\0') {
		if (*str == c) {
			++num;
		}
		++str;
	}

	return num;
}

int LatentRelevance::parse_vector(char* buf, float* vector)
{
	int index = 0;
	char* ptr = strtok(buf, " ");
	float mod = 0.0f;

	while (ptr != NULL) {
		vector[index++] = strtof(ptr, NULL);
		mod += fabs(vector[index-1]);
		ptr = strtok(NULL, " ");
	}

	// Check zero vector
	if (mod < 1e-6) {
		printf("[WARNING] Zero vector!\n");
		return -1;
	}

	return index;
}

int LatentRelevance::normalize_vector(float* vector, int size)
{
	float l2_norm = 0;
	for (int i = 0; i < size; ++i) {
		l2_norm += vector[i] * vector[i];
	}
	float mod = sqrt(l2_norm);

	for (int i = 0; i < size; ++i) {
		vector[i] = vector[i] / mod;
	}

	return 0;
}

int LatentRelevance::initialize()
{
	// Initialize w_cos and w0
	w_cos = 4.0f;
	w0 = 0.0f;

	// Intialize v
	if (factorize_mode == 1) {
		for (int i = 0; i < 2 * vector_size * k; ++i) {
			srand(static_cast<unsigned int>(time(NULL) + i));
			v[i] = 0.1 * (static_cast<float>(rand()) / RAND_MAX - 0.5);
		}
	}
	
	return 0;
}

int LatentRelevance::pre_calculate()
{
	// Pre-calculate intermediate variables for speed-up
	for (int m = 0; m < data_num; ++m) {
		// Normalize x1 and x2 for all data
		normalize_vector(m_data[m].x, vector_size);
		normalize_vector(m_data[m].x + vector_size, vector_size);

		// Calculate cosine(x1, x2)
		m_data[m].cos = 0.0f;
		for (int j = 0; j < vector_size; ++j) {
			m_data[m].cos += m_data[m].x[j] * m_data[m].x[j+vector_size];
		}

		// Calculate x_ij
		if (factorize_mode == 0) {
			for (int i = 0; i < vector_size; ++i) {
				for (int j = 0; j < vector_size; ++j) {
					m_data[m].x_ij[i*vector_size+j] = m_data[m].x[i] * m_data[m].x[j+vector_size];
				}
			}
		}
	}

	return 0;
}

int LatentRelevance::train()
{
	if (initialize() != 0) {
		printf("[ERROR] Initialize failed!\n");
		return -1;
	}
	
	if (pre_calculate() != 0) {
		printf("[ERROR] Pre-calculate failed!\n");
		return -1;
	}

	if (run_gd() != 0) {
		printf("[ERROR] Running gradient descent failed!\n");
		return -1;
	}

	return 0;
}

int LatentRelevance::run_gd()
{
	int iter_num = 1;
	float loss = 0.0f;
	loss = calculate_loss();

	// Save the gradients of weights for line search
	float* gradient_weights = NULL;
	if (factorize_mode == 0) {
		gradient_weights = new float[vector_size*vector_size];
	}
	else {
		gradient_weights = new float[2*vector_size*k];
	}

	printf("------------------------------------------------------------------------\n");
	printf("Iteration Process... [%d iterations in total]\n", MAX_STOP_ITER_NUM);
	printf("------------------------------------------------------------------------\n");
	
	// Iteration
	while (loss > MIN_STOP_LOSS * data_num && iter_num <= MAX_STOP_ITER_NUM) {
		// Calculate gradients of w0 and w_cos
		float gradient_w0 = calculate_gradient_w0();
		float gradient_wcos = calculate_gradient_wcos();

		// calculate gradients w or v
		if (factorize_mode == 0) {
			for (int i = 0; i < vector_size; ++i) {
				for (int j = 0; j < vector_size; ++j) {
					int index = i * vector_size + j;
					gradient_weights[index] = calculate_gradient_w(i, j);
				}
			}
		}
		else if (factorize_mode == 1) {
            // Store intermediate gradient variables
            float* temp_gd1 = new float[data_num*k];
            float* temp_gd2 = new float[data_num*k];
            
			for (int j = 0; j < k; ++j) {
                // Pre-calculate temp_gd
                for (int m = 0; m < data_num; ++m) {
                    int index_temp_gd = j * data_num + m;
                    temp_gd1[index_temp_gd] = 0;
                    temp_gd2[index_temp_gd] = 0;
					
                    for (int i = 0; i < vector_size; ++i) {
						temp_gd1[index_temp_gd] += v[i*k+j] * m_data[m].x[i];
					}

					for (int i = vector_size; i < 2 * vector_size; ++i) {
						temp_gd2[index_temp_gd] += v[i*k+j] * m_data[m].x[i];
					}
				}     

                for (int i = 0; i < 2 * vector_size; ++i) {
                    int index = i * k + j;
                    if (i < vector_size) {
                        gradient_weights[index] = calculate_gradient_v(i, j, temp_gd2);
                    }
                    else {
                        gradient_weights[index] = calculate_gradient_v(i, j, temp_gd1);
                    }
                }
			}

			delete temp_gd1;
			temp_gd1 = NULL;
			delete temp_gd2;
			temp_gd2 = NULL;
		}
		
		printf("Iter[%d]\tLoss[%f]\tW0[%f]\tWCos[%f]\n", iter_num, loss, w0, w_cos);

		// Find next point by line search
		line_search(gradient_w0, gradient_wcos, gradient_weights, loss);
		++iter_num;
	}

	delete gradient_weights;
	gradient_weights = NULL;

	return 0;
}

float LatentRelevance::calculate_gradient_w0()
{
	float gradient = 0.0f;
	for (int i = 0; i < data_num; ++i) {
		gradient += -1 * (1 - m_data[i].score);
		if (m_data[i].y <= 0) {
			gradient += 1.0;
		}
	}

	// L2 regularization
	gradient += 2 * c * w0;

	return gradient;
}

float LatentRelevance::calculate_gradient_wcos()
{
	float gradient = 0.0f;
	for (int i = 0; i < data_num; ++i) {
		gradient += -1 * (1 - m_data[i].score) * m_data[i].cos;
		if (m_data[i].y <= 0) {
			gradient += m_data[i].cos;
		}
	}

	// L2 regularization
	gradient += 2 * c * w_cos;
	
	return gradient;
}

float LatentRelevance::calculate_gradient_w(int index1, int index2)
{
	float gradient = 0.0f;
	int index = index1 * vector_size + index2;
	for (int m = 0; m < data_num; ++m) {
		// Filter irrelevant samples
		if (fabs(m_data[m].x[index1]) < 1e-6 || fabs(m_data[m].x[index2+vector_size]) < 1e-6) {
			continue;
		}

		gradient += (m_data[m].score - 1) * m_data[m].x_ij[index];
		if (m_data[m].y <= 0) {
			gradient += m_data[m].x_ij[index];
		}
	}

	// L2 regularization
	gradient += 2 * c * w[index];
	
	return gradient;
}

float LatentRelevance::calculate_gradient_v(int index1, int index2, float* temp_gd)
{
	float gradient = 0.0f;
	for (int m = 0; m < data_num; ++m) {
		if (fabs(m_data[m].x[index1]) < 1e-6) {
			continue;
		}

        int index_temp_gd = index2 * data_num + m;

		// Caculate gradients of factorization variables
		float factorize_gd = 0.0f;

		if (index1 < vector_size) {
			factorize_gd -= 0.5 * v[(index1+vector_size)*k+index2] * m_data[m].x[index1] * m_data[m].x[index1+vector_size];
		}
		else {
			factorize_gd -= 0.5 * v[(index1-vector_size)*k+index2] * m_data[m].x[index1] * m_data[m].x[index1-vector_size];
		}
		
		factorize_gd += temp_gd[index_temp_gd] * m_data[m].x[index1];	
		
		gradient += (m_data[m].score - 1) * factorize_gd;
		if (m_data[m].y <= 0) {
			gradient += factorize_gd;
		}
	}

	// L2 regularization
	gradient += 2 * c * v[index1*k+index2];
	
	return gradient;
}

int LatentRelevance::line_search(float gradient_w0, float gradient_wcos, float* gradient_weights, float& loss)
{
	const float C1 = 1e-4;
	const float backtrack_ratio = 0.2f;

	// Save original variable values
	float orig_w0 = w0;
	float orig_wcos = w_cos;
	float orig_loss = loss;

	// Save original weights
	float* orig_weights = NULL;
	float* weights = NULL;
	int weights_size = 0;
	if (factorize_mode == 0) {
		weights_size = vector_size * vector_size;
		weights = w;
	}
	else {
		weights_size = 2 * vector_size * k;
		weights = v;
	}	
	orig_weights = new float[weights_size];
	memcpy(orig_weights, weights, weights_size * sizeof(float));

	const float GRAD_SHRINK_FACTOR = 0.1f;
	gradient_w0 /= data_num * GRAD_SHRINK_FACTOR;
	gradient_wcos /= data_num * GRAD_SHRINK_FACTOR;

	// For steepest gradient descent, descent direction is -gradient
	float dir_grad = 0.0f;
	dir_grad -= gradient_w0 * gradient_w0 + gradient_wcos * gradient_wcos;
	for (int i = 0; i < weights_size; ++i) {
		dir_grad -= gradient_weights[i] * gradient_weights[i];
	}

	// Set initial step_size
	float step_size = learn_rate;
//	if (pre_loss_step < -1e-6) {
//		step_size = pre_loss_step / dir_grad;
//	}

	// Backtracking Line Search
	while (step_size > 1e-9) {
		// Get next point
		w0 = orig_w0 - step_size * gradient_w0;
		w_cos = orig_wcos - step_size * gradient_wcos;
//        printf("%f\t%f\t%f\t%f\n", w0, w_cos, gradient_w0, gradient_wcos);
//        getchar();
		
        for (int i = 0; i < weights_size; ++i) {
			weights[i] = orig_weights[i] - step_size * gradient_weights[i];
		}

		loss = calculate_loss();
//		printf("%f\t%f\t%F\t%f\t%f\n", loss, orig_loss, orig_loss + C1 * step_size * dir_grad, step_size, dir_grad);
//      getchar();
        
		// Wolfe condition
		if (loss <= orig_loss + C1 * step_size * dir_grad) {
			pre_loss_step = step_size * dir_grad;
			break;
		}

		step_size *= backtrack_ratio;
	}

	// Back to original point if no avaliable step_size
	if (loss >= orig_loss) {
		w0 = orig_w0;
		w_cos = orig_wcos;
		memcpy(weights, orig_weights, weights_size * sizeof(float));
		loss = orig_loss;
	}

	delete orig_weights;
	orig_weights = NULL;

	return 0;
}

float LatentRelevance::proximal_operator_L1(float weight)
{
	float t = c * learn_rate;
	if (weight >= t) {
		return weight - t;
	}
	else if (weight <= -t) {
		return weight + t;
	}
	else {
		return 0;
	}
}

float LatentRelevance::calculate_loss()
{
	// Calculate scores for all data
	for (int i = 0; i < data_num; ++i) {
		m_data[i].score = predict(m_data + i);
	}	

	// Calculate loss
	float loss = 0.0f;
	for (int i = 0; i < data_num; ++i) {
		loss -= log(m_data[i].score);
		if (m_data[i].y <= 0) {
			loss -= log(1 / (m_data[i].score + 1e-6) - 1);
		}
	}

	// Regularization terms
	float reg_loss = 0.0;
	reg_loss += w0 * w0;
	reg_loss += w_cos * w_cos;
	if (factorize_mode == 0) {
		for (int i = 0; i < vector_size * vector_size; ++i) {
			reg_loss += w[i] * w[i];
		}
	}
	else {
		for (int i = 0; i < 2 * vector_size * k; ++i) {
			reg_loss += v[i] * v[i];
		}
	}
	loss += reg_loss * c;

	return loss;
}

int LatentRelevance::save_model(const char* model_name)
{
	FILE* fp = fopen(model_name, "w");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Saving model failed!\n", model_name);
		return -1;
	}

	fprintf(fp, "%d\n%d\n%d\n%f\n%f\n", factorize_mode, vector_size, k, w0, w_cos);
	if (factorize_mode == 0) {
		for (int i = 0; i < vector_size * vector_size; ++i) {
			fprintf(fp, "%f\n", w[i]);
		}
	}
	else if (factorize_mode == 1) {
		for (int i = 0; i < 2 * vector_size * k; ++i) {
			fprintf(fp, "%f\n", v[i]);
		}
	}

	fclose(fp);
}

int LatentRelevance::load_model(const char* model_name)
{
	FILE* fp = fopen(model_name, "r");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Loading model failed!\n", model_name);
		return -1;
	}

	const int MAX_MODEL_LINE_LEN = 1024;
	char buf[MAX_MODEL_LINE_LEN];
	int i = 0;

	while (fgets(buf, MAX_MODEL_LINE_LEN, fp) != NULL) {
		if (i == 0) {
			factorize_mode = atoi(buf);
			if (factorize_mode != 0 && factorize_mode != 1) {
				printf("[ERROR] Invalid factorize mode!\n");
				fclose(fp);
				return -1;
			}
		}
		else if (i == 1) {
			vector_size = atoi(buf);
		}
		else if (i == 2) {
			k = atoi(buf);
		}
		else if (i == 3) {
			w0 = strtof(buf, NULL);
		}
		else if (i == 4) {
			w_cos = strtof(buf, NULL);
		}
		else {								// Load w or v
			if (factorize_mode == 0) {
				// Allocate w
				if (w == NULL) {
					w = new float[vector_size*vector_size];
				}

				w[i-5] = strtof(buf, NULL);
				if (i - 5 > vector_size * vector_size) {
					break;
				}
			}
			else if (factorize_mode == 1) {
				if (v == NULL) {
					v = new float[2*vector_size*k];
				}

				v[i-5] = strtof(buf, NULL);
				if (i - 5 > 2 * vector_size * k) {
					break;
				}
			}
		}

		++i;
	}

	fclose(fp);
}

int LatentRelevance::test(const char* test_file_name, const char* model_name)
{
	if (load_model(model_name) != 0) {
		printf("[ERROR] Load model failed!\n");
		return -1;
	}

	if (read_data(test_file_name) != 0) {
		printf("[ERROR] Read data failed!\n");
		return -1;
	}
	
	if (pre_calculate() != 0) {
		printf("[ERROR] Pre-calculate failed!\n");
		return -1;
	}

	const int MAX_FILE_NAME_LEN = 1024;
	char res_file_name[MAX_FILE_NAME_LEN];
	snprintf(res_file_name, MAX_FILE_NAME_LEN, "%s.res", test_file_name);

	FILE* fp = fopen(res_file_name, "w");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Testing file failed!\n", res_file_name);
		return -1;
	}

	for (int i = 0; i < data_num; ++i) {
		float score = predict(m_data + i);
		int label = m_data[i].y;
		if (label < 0) {				// Change -1 to 0 for calculating AUC
			label = 0;
		}

		fprintf(fp, "%f\t1\t%d\n", score, label);
	}

	printf("[NOTICE] Predict results are saved in %s\n", res_file_name);
	fclose(fp);

	return 0;
}

float LatentRelevance::predict(data_t* data)
{
	// Calculate latent score
	float lat_score = 0.0f;

	if (factorize_mode == 0) {
		// Calculate sigma(w_ij * x1_i * x2_j)
		for (int i = 0; i < vector_size; ++i) {
			if (fabs(data->x[i]) < 1e-6) {
				continue;
			}
			for (int j = 0; j < vector_size; ++j) {
				if (fabs(data->x[j+vector_size]) < 1e-6 || i == j) {
					continue;
				}
			
				int index = i * vector_size + j;
				lat_score += w[index] * data->x_ij[index];
			}
		}
	}
	else if (factorize_mode == 1) {
		// Calculate sigma(<v_i, v_j>  * x1_i * x2_j) (i != j)
		for (int f = 0; f < k; ++f) {
			float temp_score1 = 0.0f;
			float temp_score2 = 0.0f;
			float temp_score3 = 0.0f;
			for (int i = 0; i < 2 * vector_size; ++i) {
				if (fabs(data->x[i]) < 1e-6) {
					continue;
				}

				// Calculate factorization score
				if (i < vector_size) {
					temp_score1 += v[i*k+f] * data->x[i];
					temp_score3 += v[i*k+f] * v[(i+vector_size)*k+f] * data->x[i] * data->x[i+vector_size];
				}
				else {
					temp_score2 += v[i*k+f] * data->x[i];
				}
			}

			float temp_score4 = temp_score1 + temp_score2;
			lat_score += temp_score4 * temp_score4 - temp_score1 * temp_score1 - temp_score2 * temp_score2 - temp_score3;
		}

		lat_score *= 0.5;
	}
	else {
		printf("[ERROR] Invalid factorize mode (should be 0 or 1)!\n");
	}

	// Logistic transform
	float score = w_cos * data->cos + w0 + lat_score;
	const float MAX_POWER_EXP = 10.0f;
	if (score > MAX_POWER_EXP) {
		score = MAX_POWER_EXP;
	}
	else if (score < -MAX_POWER_EXP){
		score = -MAX_POWER_EXP;
	}

	score = 1 / (1 + exp(-score));

	return score;
}

} // namespace latent_relevance

