// Copyright (c) 2014 Baidu Corporation
// @file:   train.cpp
// @brief:  tool for training and testing latent relevance
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-07-17

#include "latent_relevance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_FILE_NAME_LEN = 1024;

// Function declaration
void print_help();
int parse_command_line(latent_relevance::LatentRelevance* lat_rel, int argc, char **argv, char *train_file, char *model_file);

int main(int argc, char** argv)
{
	char train_file[MAX_FILE_NAME_LEN];
	char model_file[MAX_FILE_NAME_LEN];
	latent_relevance::LatentRelevance* lat_rel = new latent_relevance::LatentRelevance(); 

	if (parse_command_line(lat_rel, argc, argv, train_file, model_file) != 0) {
		print_help();
		return -1;
	}
//	printf("%d\t%d\t%f\t%f\t%s\t%s\n", lat_rel->factorize_mode, lat_rel->k, lat_rel->c, lat_rel->learn_rate, train_file, model_file);

	lat_rel->read_data(train_file);
	lat_rel->train();
	lat_rel->save_model(model_file);

	delete lat_rel;

	return 0;
}

// Print help information
void print_help()
{
	printf(
		"Usage: ./train [options] training_file [model_file]\n"
		"options:\n"
		"	-f mode: 1 with factorization, 0 no factorizatoin (default 1)\n"
		"	-k factorization dimension (default 3)\n"
		"	-c regularization coefficient (default 0.1)\n"
        "   -l learning rate (default 1.0)\n"
		"training_file format: \n"
		"	label \\t x1 \\t x2 (example: 1 \\t 0.1 0.2 \\t 0.3 0.1)\n"
	);
	
	return;
}

// Parse command 
int parse_command_line(latent_relevance::LatentRelevance* lat_rel, int argc, char **argv, char *train_file, char *model_file)
{
	// Set default parameters
	lat_rel->set_factorize_mode(1);
	lat_rel->set_k(3);
	lat_rel->set_c(0.1);
	lat_rel->set_learn_rate(1.0);
	
	// parse options
	int i = 0;
	for (i = 1; i < argc; ++i) {
		if (argv[i][0] != '-') {
			break;
		}

		if (++i >= argc) {
			return -1;
		}

		switch (argv[i-1][1]) {
			case 'f':
				int factorize_mode = atoi(argv[i]);
				if (factorize_mode != 0 && factorize_mode != 1) {
					printf("[ERROR] Invalid -f value (should be 0 or 1)!\n");
					return -1;
				}
				lat_rel->set_factorize_mode(factorize_mode);
				break;

			case 'k':
				int k = atoi(argv[i]);
				if (k <= 0) {
					printf("[ERROR] Invalid -k value (should be > 0)!\n");
					return -1;
				}
				lat_rel->set_k(k);
				break;

			case 'c':
				float c = atof(argv[i]);
				if (c < 0) {
					printf("[ERROR] Invalid -c value (should be > 0)!\n");
					return -1;
				}
				lat_rel->set_c(c);
				break;

			case 'l':
				float learn_rate = atof(argv[i]);
				if (learn_rate < 0) {
					printf("[ERROR] Invalid -l value (should be > 0)\n");
					return -1;
				}				
				lat_rel->set_learn_rate(learn_rate);
				break;
			
			default:
				printf("[ERROR] Unknown option: -%c\n", argv[i-1][1]);
				return -1;
		}
	}

	if (i >= argc) {
		return -1;
	}

	// Parse input file name
	snprintf(train_file, MAX_FILE_NAME_LEN, "%s", argv[i]);

	// Parse model file
	if (i < argc - 1) {
		snprintf(model_file, MAX_FILE_NAME_LEN, "%s", argv[i+1]);
	}
	else {						// Default model file name
		char* ptr = strrchr(argv[i], '/');
		if (ptr == NULL) {
			ptr = argv[i];		// argv[i] is the input file name
		}
		else {
			++ptr;				// Ignor path
		}
		sprintf(model_file, "%s.model", ptr);
	}

	return 0;
}

