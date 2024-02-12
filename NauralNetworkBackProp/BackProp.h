#pragma once
#include<iostream>
#include<assert.h>
#include<stdio.h>
#include<cmath>
#include <time.h>
#include <stdlib.h>
using namespace std;

class BackProp
{
	//	output of each neuron
	double** out;
	//	delta error value for each neuron
	double** delta;
	//	vector of weights for each neuron
	double*** weight;
	//	no of layers in net
	//	including input layer
	int numl;
	//	vector of numl elements for size 
	//	of each layer
	int* layerSize;
	//	learning rate
	double learningRate;
	//	momentum parameter
	double momentum;
	//	storage for weight-change made
	//	in previous epoch
	double*** prevDwt;
	//	squashing function
	double sigmoid(double in);
public:
	~BackProp();
	//	initializes and allocates memory
	BackProp(int nl, int* sz, double b, double a);
	//	backpropogates error for one set of input
	void backPropagate(double* in, double* tgt);
	//	feed forwards activations for one set of inputs
	void feedForward(double* in);
	//	returns mean square error of the net
	double mse(double* tgt) const;
	//	returns i'th output of the net
	double Out(int i) const;
};