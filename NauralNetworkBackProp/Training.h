#pragma once
#include "BackProp.h"
#include <omp.h>
#include <ctime>
using namespace std;
class Training
{
public:
	void start();
	Training();
	~Training();
};