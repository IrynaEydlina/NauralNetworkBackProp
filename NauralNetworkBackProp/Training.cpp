#include "Training.h"

Training::Training()
{
}


Training::~Training()
{
}

void Training::start()
{
	const int length = 999;
	double Num, Num2;
	double data[length][3];
	for (int i = 0; i < length; i++)
	{
		Num = (double)rand() / RAND_MAX;
		data[i][0] = Num;
		Num2 = (double)rand() / RAND_MAX;
		data[i][1] = Num2;
		data[i][2] = Num * Num2;
	}
	srand(time(NULL));
	double testData[4][2];
	double result[4];
	for (int i = 0; i < 4; i++)
	{
		Num = (double)rand() / RAND_MAX;
		testData[i][0] = Num;
		Num2 = (double)rand() / RAND_MAX;
		testData[i][1] = Num2;
		result[i] = Num * Num2;
	}
	int numLayers = 5, lSz[] = { 2, 10, 5, 2, 1 };
	double beta = 0.1, alpha = 0.1, Thresh = 0.0000000001;
	long num_iter = 200000000;

	BackProp* bp = new BackProp(numLayers, lSz, beta, alpha);
	int countRight = 0;
	int maxRight = length;
	unsigned int start_time = clock();
	long i;
	//#pragma omp parallel num_threads(4)//4 threads (number of cores involved)
	//{
		//#pragma omp for					 
	for (i = 0; i < num_iter; i++)
	{
		bp->backPropagate(data[i % length], &data[i % length][2]);
		if (bp->mse(&data[i % length][2]) < Thresh)
		{
			countRight++;
			if (countRight == maxRight)
				break;
		}
	}
	//}
	unsigned int end_time = clock();
	unsigned int search_time = (end_time - start_time);
	cout << search_time << " ms" << endl;
	/*	if (i == num_iter)
			cout << endl << i << " iterations" << endl
			<< "MSE: " << bp->mse(&data[(i - 1) % length][2]) << endl;*/

	for (long i = 0; i < 4; i++)
	{
		bp->feedForward(testData[i]);
		cout << testData[i][0] << "  " << testData[i][1] << "  " << bp->Out(0) << "|" << result[i] << endl;
	}
}
