#include "Training.h"
int main()
{
	setlocale(LC_ALL, "en-US");
	Training trainer;
	trainer.start();
	/*unsigned int start_time = clock();
	long dst[1000];
	long i;
	#pragma omp parallel num_threads(3)
	{
	#pragma omp for
		for (i = 0; i < 2000000000; i++)
		{
			dst[i%1000] = sqrt(i);
			if (i == 2000000000 - 1)
				cout << "test" << endl;
		}
	}
	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	cout << search_time << endl;*/
	return 0;
}