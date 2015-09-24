// Copyright (c) 2014 Baidu Corporation
// @file:   test.cpp
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
int parse_command_line(int argc, char **argv, char *test_file, char *model_file);

int main(int argc, char** argv)
{
	char test_file[MAX_FILE_NAME_LEN];
	char model_file[MAX_FILE_NAME_LEN];
	latent_relevance::LatentRelevance* lat_rel = new latent_relevance::LatentRelevance(); 

	if (parse_command_line(argc, argv, test_file, model_file) != 0) {
		print_help();
		return -1;
	}
//	printf("%d\t%d\t%f\t%f\t%s\t%s\n", lat_rel->factorization, lat_rel->k, lat_rel->c, lat_rel->learn_rate, train_file, model_file);

	lat_rel->test(test_file, model_file);

	delete lat_rel;

	return 0;
}

// Print help information
void print_help()
{
	printf(
		"Usage: ./test test_file model_file\n"
		"test_file format: label \\t x1 \\t x2 (example: 1 \\t 0.1 0.2 \\t 0.3 0.1)\n"
	);
	
	return;
}

// Parse command 
int parse_command_line(int argc, char **argv, char *test_file, char *model_file)
{
	if (argc != 3) {
		return -1;
	}

	snprintf(test_file, MAX_FILE_NAME_LEN, "%s", argv[1]);
	snprintf(model_file, MAX_FILE_NAME_LEN, "%s", argv[2]);

	return 0;
}

