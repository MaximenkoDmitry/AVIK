#pragma once
#include <iostream>
#include <chrono>
#include <utility>

#include <helper_image.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_PER_BLOCK_X  32
#define THREAD_PER_BLOCK_Y  32

#define MATRIX_BLOCK_WIDTH (1024 * 16)
#define MATRIX_BLOCK_HEIGHT (1024 * 16)

using namespace std;
