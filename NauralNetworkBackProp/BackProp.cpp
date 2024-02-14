#include "BackProp.h"

BackProp::BackProp(int nl, int* sz, double b, double a) :learningRate(b), momentum(a)
{
	numl = nl;
	layerSize = new int[numl];
	int i = 0;
	for (i = 0; i < numl; i++)
	{
		layerSize[i] = sz[i];
	}
	out = new double* [numl];
	for (i = 0; i < numl; i++)
	{
		out[i] = new double[layerSize[i]];
	}
	delta = new double* [numl];
	for (i = 1; i < numl; i++)
	{
		delta[i] = new double[layerSize[i]];
	}
	weight = new double** [numl];
	for (i = 1; i < numl; i++)
	{
		weight[i] = new double* [layerSize[i]];
	}
	for (i = 1; i < numl; i++)
	{
		for (int j = 0; j < layerSize[i]; j++)
		{
			weight[i][j] = new double[layerSize[i - 1] + 1];
		}
	}
	prevDwt = new double** [numl];
	for (i = 1; i < numl; i++)
	{
		prevDwt[i] = new double* [layerSize[i]];
	}
	for (i = 1; i < numl; i++)
	{
		for (int j = 0; j < layerSize[i]; j++)
		{
			prevDwt[i][j] = new double[layerSize[i - 1] + 1];
		}
	}
	srand((unsigned)(time(NULL)));
	for (i = 1; i < numl; i++)
		for (int j = 0; j < layerSize[i]; j++)
			for (int k = 0; k < layerSize[i - 1] + 1; k++)
				weight[i][j][k] = (double)(rand()) / (RAND_MAX / 2) - 1;
	for (i = 1; i < numl; i++)
		for (int j = 0; j < layerSize[i]; j++)
			for (int k = 0; k < layerSize[i - 1] + 1; k++)
				prevDwt[i][j][k] = (double)0.0;
}

BackProp::~BackProp()
{
	int i = 0;
	for (i = 0; i < numl; i++)
		delete[] out[i];
	delete[] out;
	for (i = 1; i < numl; i++)
		delete[] delta[i];
	delete[] delta;
	for (i = 1; i < numl; i++)
		for (int j = 0; j < layerSize[i]; j++)
			delete[] weight[i][j];
	for (i = 1; i < numl; i++)
		delete[] weight[i];
	delete[] weight;
	for (i = 1; i < numl; i++)
		for (int j = 0; j < layerSize[i]; j++)
			delete[] prevDwt[i][j];
	for (i = 1; i < numl; i++)
		delete[] prevDwt[i];
	delete[] prevDwt;
	delete[] layerSize;
}

double BackProp::sigmoid(double in)
{
	return (double)(1 / (1 + exp(-in)));
}
double BackProp::mse(double* tgt) const
{
	double mse = 0;
	for (int i = 0; i < layerSize[numl - 1]; i++)
	{
		mse += (tgt[i] - out[numl - 1][i]) * (tgt[i] - out[numl - 1][i]);
	}
	return mse / 2;
}

double BackProp::Out(int i) const
{
	return out[numl - 1][i];
}

void BackProp::feedForward(double* in)
{
	double sum;
	int i = 0;
	for (int i = 0; i < layerSize[0]; i++)
		out[0][i] = in[i];

	for (i = 1; i < numl; i++)
	{
		for (int j = 0; j < layerSize[i]; j++)
		{
			sum = 0.0;
			for (int k = 0; k < layerSize[i - 1]; k++)
			{
				sum += out[i - 1][k] * weight[i][j][k];
			}
			sum += weight[i][j][layerSize[i - 1]];
			out[i][j] = sigmoid(sum);
		}
	}
}

void BackProp::backPropagate(double* in, double* tgt)
{
	double sum;
	feedForward(in);
	int i = 0;
	for (int i = 0; i < layerSize[numl - 1]; i++)
	{
		delta[numl - 1][i] = out[numl - 1][i] *
			(1 - out[numl - 1][i]) * (tgt[i] - out[numl - 1][i]);
	}
	for (i = numl - 2; i > 0; i--) {
		for (int j = 0; j < layerSize[i]; j++)
		{
			sum = 0.0;
			for (int k = 0; k < layerSize[i + 1]; k++)
			{
				sum += delta[i + 1][k] * weight[i + 1][k][j];
			}
			delta[i][j] = out[i][j] * (1 - out[i][j]) * sum;
		}
	}
	for (i = 1; i < numl; i++)
	{
		for (int j = 0; j < layerSize[i]; j++)
		{
			for (int k = 0; k < layerSize[i - 1]; k++)
			{
				weight[i][j][k] += momentum * prevDwt[i][j][k];
			}
			weight[i][j][layerSize[i - 1]] += momentum * prevDwt[i][j][layerSize[i - 1]];
		}
	}
	for (i = 1; i < numl; i++) {
		for (int j = 0; j < layerSize[i]; j++)
		{
			for (int k = 0; k < layerSize[i - 1]; k++)
			{
				prevDwt[i][j][k] = learningRate * delta[i][j] * out[i - 1][k];
				weight[i][j][k] += prevDwt[i][j][k];
			}
			prevDwt[i][j][layerSize[i - 1]] = learningRate * delta[i][j];
			weight[i][j][layerSize[i - 1]] += prevDwt[i][j][layerSize[i - 1]];
		}
	}
}