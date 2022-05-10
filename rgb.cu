#include "header.cuh"

__global__ void KernelBlurFilterRGB(unsigned char* dst, unsigned char* src, int height, int width, int s_pitch, int d_pitch) {

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_center = d_pitch * thread_y + thread_x;
    int src_center = s_pitch * (thread_y + 1) + thread_x + 3;
    int src_low = s_pitch * thread_y + thread_x + 3;
    int src_high = s_pitch * (thread_y + 2) + thread_x + 3;
    int pixel = 0;
    if (thread_x < width * 3 && thread_y < height) {
        pixel = src[src_high - 3] + src[src_high] + src[src_high + 3];
        pixel += src[src_center - 3] + 2 * src[src_center] + src[src_center + 3];
        pixel += src[src_low - 3] + src[src_low] + src[src_low + 3];
        pixel /= 10;
        dst[dst_center] = (unsigned char)pixel;
    }
}

__global__ void KernelBlurFilterRGBOptimized(uint32_t* dst, uint32_t* src, int height, int width, int s_pitch, int d_pitch) {
    __shared__ unsigned char mem[3 * (4 * THREAD_PER_BLOCK_X + 8) * (THREAD_PER_BLOCK_Y + 2)];

    int thread_x = 3 * blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_center = d_pitch * thread_y + thread_x;
    int src_center = s_pitch * (thread_y + 1) + thread_x + 128 / 4;
    int src_low = s_pitch * thread_y + thread_x + 128 / 4;
    int src_high = s_pitch * (thread_y + 2) + thread_x + 128 / 4;

    int mem_center = (threadIdx.y + 1) * 3 * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 12;
    int mem_low = threadIdx.y * 3 * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 12;
    int mem_high = (threadIdx.y + 2) * 3 * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 12;

    int mem32_center = mem_center / 4;
    int mem32_low = mem_low / 4;
    int mem32_high = mem_high / 4;

    uint32_t* mem32 = (uint32_t*)mem;

    if (thread_x * 4 < 12 + width * 3 && thread_y <= height) {
        mem32[mem32_center] = src[src_center];
        mem32[mem32_center + THREAD_PER_BLOCK_X] = src[src_center + THREAD_PER_BLOCK_X];
        mem32[mem32_center + 2 * THREAD_PER_BLOCK_X] = src[src_center + 2 * THREAD_PER_BLOCK_X];

        __syncthreads();

        if (threadIdx.x == 0) {
            mem32[mem32_center - 1] = src[src_center - 1];
            mem32[mem32_high - 1] = src[src_high - 1];
            mem32[mem32_low - 1] = src[src_low - 1];
        }
        if (threadIdx.x == blockDim.x - 1) {
            mem32[mem32_center + 1 + 2 * THREAD_PER_BLOCK_X] = src[src_center + 1 + 2 * THREAD_PER_BLOCK_X]; 
            mem32[mem32_high + 1 + 2 * THREAD_PER_BLOCK_X] = src[src_high + 1 + 2 * THREAD_PER_BLOCK_X];
            mem32[mem32_low + 1 + 2 * THREAD_PER_BLOCK_X] = src[src_low + 1 + 2 * THREAD_PER_BLOCK_X];
        }

        if (threadIdx.y == 0) {
            mem32[mem32_low] = src[src_low];
            mem32[mem32_low + THREAD_PER_BLOCK_X] = src[src_low + THREAD_PER_BLOCK_X];
            mem32[mem32_low + 2 * THREAD_PER_BLOCK_X] = src[src_low + 2 * THREAD_PER_BLOCK_X];
        }
        if (threadIdx.y == blockDim.y - 1) {
            mem32[mem32_high] = src[src_high];
            mem32[mem32_high + THREAD_PER_BLOCK_X] = src[src_high + THREAD_PER_BLOCK_X];
            mem32[mem32_high + 2 * THREAD_PER_BLOCK_X] = src[src_high + 2 * THREAD_PER_BLOCK_X];
        }

        __syncthreads();

        for (int i = 0; i < 3; i++) {
            uint32_t res = 0;
            for (int j = 0; j < 4; j++) {
                int32_t pixel = 0;
                pixel = mem[mem_high - 3 + j + 4 * i * THREAD_PER_BLOCK_X] + mem[mem_high + j + 4 * i * THREAD_PER_BLOCK_X] + mem[mem_high + 3 + j + 4 * i * THREAD_PER_BLOCK_X];
                pixel += mem[mem_center - 3 + j + 4 * i * THREAD_PER_BLOCK_X] + 2 * mem[mem_center + j + 4 * i * THREAD_PER_BLOCK_X] + mem[mem_center + 3 + j + 4 * i * THREAD_PER_BLOCK_X];
                pixel += mem[mem_low - 3 + j + 4 * i * THREAD_PER_BLOCK_X] + mem[mem_low + j + 4 * i * THREAD_PER_BLOCK_X] + mem[mem_low + 3 + j + 4 * i * THREAD_PER_BLOCK_X];
                pixel /= 10;
                res |= (pixel << (8 * j));
            }
            dst[dst_center + i * THREAD_PER_BLOCK_X] = res;
        }
    }

}

double gpuBlurFilterRGB(unsigned char* host_dst, unsigned char* host_src, int width, int height) {
    cudaError_t cudaStatus = cudaSuccess;
    unsigned char* dev_src = nullptr;
    unsigned char* dev_dst = nullptr;
    size_t s_pitch = 0;
    size_t d_pitch = 0;
    int channels = 3;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaMallocPitch((void**)&dev_src, &s_pitch, channels * (2 + MATRIX_BLOCK_WIDTH) * sizeof(unsigned char), 2 + MATRIX_BLOCK_HEIGHT);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMallocPitch failed!\n";
        return -1;
    }

    cudaStatus = cudaMallocPitch((void**)&dev_dst, &d_pitch, channels * MATRIX_BLOCK_WIDTH * sizeof(unsigned char), MATRIX_BLOCK_HEIGHT);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMallocPitch failed!\n";
        cudaFree(dev_src);
        return -1;
    }

    cudaEventRecord(start, 0);

    int max_i = (height + MATRIX_BLOCK_HEIGHT - 1) / MATRIX_BLOCK_HEIGHT;
    int max_j = (width + MATRIX_BLOCK_WIDTH - 1) / MATRIX_BLOCK_WIDTH;
    for (int i = 0; i < max_i; i++) {
        for (int j = 0; j < max_j; j++) {

            int block_width = MATRIX_BLOCK_WIDTH;
            int block_height = MATRIX_BLOCK_HEIGHT;

            if (block_width * j + block_width > width) {
                block_width = width - block_width * j;
            }

            if (block_height * i + block_height > height) {
                block_height = height - block_height * i;
            }
            int host_src_index = i * MATRIX_BLOCK_HEIGHT * width * channels + MATRIX_BLOCK_WIDTH * j * channels;
            int dev_src_column_index = 1 * channels;
            int dev_src_row_index = (int)s_pitch;
            int read_block_width = block_width * channels;
            int read_block_height = block_height;
            if (j != 0) {
                dev_src_column_index -= channels;
                host_src_index -= channels;
                read_block_width += channels;
            }
            if (i != 0) {
                dev_src_row_index -= (int)s_pitch;
                host_src_index -= width * channels;
                read_block_height++;
            }
            if (i != max_i - 1) {
                read_block_height++;
            }
            if (j != max_j - 1) {
                read_block_width += channels;
            }

            cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index + dev_src_row_index, s_pitch, host_src + host_src_index, width * channels * sizeof(char), read_block_width * sizeof(char), read_block_height, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync1 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
            if (i == 0) {
                cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index, s_pitch, host_src + host_src_index, width * channels * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync2 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (i == max_i - 1) {
                if (i != 0) {
                    host_src_index += width * channels;
                }
                cudaStatus = cudaMemcpy2DAsync(dev_src + s_pitch * (block_height + 1) + dev_src_column_index, s_pitch, host_src + host_src_index + (block_height - 1) * width * channels, width * channels * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync3 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (j == 0) {
                for (int k = 0; k < block_height + 2; k++) {
                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k, dev_src + s_pitch * k + channels, channels, cudaMemcpyDeviceToDevice);
                    if (cudaStatus != cudaSuccess) {
                        cerr << "cudaMemcpyAsync4 failed!\n";
                        cudaFree(dev_src);
                        cudaFree(dev_dst);
                        return -1;
                    }
                }
            }
            if (j == max_j - 1) {
                for (int k = 0; k < block_height + 2; k++) {

                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + channels + block_width * channels, dev_src + s_pitch * k + block_width * channels, channels, cudaMemcpyDeviceToDevice);
                    if (cudaStatus != cudaSuccess) {
                        cerr << "cudaMemcpyAsync5 failed!\n";
                        cudaFree(dev_src);
                        cudaFree(dev_dst);
                        return -1;
                    }

                }
            }

            dim3 block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y);
            dim3 grid((block_width * channels + THREAD_PER_BLOCK_X - 1) / THREAD_PER_BLOCK_X, (block_height + THREAD_PER_BLOCK_Y - 1) / THREAD_PER_BLOCK_Y);
            
            KernelBlurFilterRGB << <grid, block >> > (dev_dst, dev_src, block_height, block_width, (int)s_pitch, (int)d_pitch);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                cerr << "KernelBlurFilter launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }

            cudaStatus = cudaMemcpy2DAsync(host_dst + i * MATRIX_BLOCK_HEIGHT * width * channels + MATRIX_BLOCK_WIDTH * j * channels, width * channels * sizeof(char), dev_dst, d_pitch, block_width * channels * sizeof(char), block_height, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync6 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

double gpuBlurFilterRGBOptimized(unsigned char* host_dst, unsigned char* host_src, int width, int height) {
    cudaError_t cudaStatus = cudaSuccess;
    unsigned char* dev_src = nullptr;
    unsigned char* dev_dst = nullptr;
    size_t s_pitch = 0;
    size_t d_pitch = 0;
    int channels = 3;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaMallocPitch((void**)&dev_src, &s_pitch, channels * (2 + MATRIX_BLOCK_WIDTH) * sizeof(unsigned char), 2 + MATRIX_BLOCK_HEIGHT);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocPitch failed!\n";
        return -1;
    }

    cudaStatus = cudaMallocPitch((void**)&dev_dst, &d_pitch, channels * MATRIX_BLOCK_WIDTH * sizeof(unsigned char), MATRIX_BLOCK_HEIGHT);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocPitch failed!\n";
        cudaFree(dev_src);
        return -1;
    }

    cudaEventRecord(start, 0);

    int max_i = (height + MATRIX_BLOCK_HEIGHT - 1) / MATRIX_BLOCK_HEIGHT;
    int max_j = (width + MATRIX_BLOCK_WIDTH - 1) / MATRIX_BLOCK_WIDTH;
    for (int i = 0; i < max_i; i++) {
        for (int j = 0; j < max_j; j++) {

            int block_width = MATRIX_BLOCK_WIDTH;
            int block_height = MATRIX_BLOCK_HEIGHT;

            if (block_width * j + block_width > width) {
                block_width = width - block_width * j;
            }

            if (block_height * i + block_height > height) {
                block_height = height - block_height * i;
            }
            int host_src_index = i * MATRIX_BLOCK_HEIGHT * width * channels + MATRIX_BLOCK_WIDTH * j * channels;
            int dev_src_column_index = 128;
            int dev_src_row_index = (int)s_pitch;
            int read_block_width = block_width * channels;
            int read_block_height = block_height;
            if (j != 0) {
                dev_src_column_index -= channels;
                host_src_index -= channels;
                read_block_width += channels;
            }
            if (i != 0) {
                dev_src_row_index -= (int)s_pitch;
                host_src_index -= width * channels;
                read_block_height++;
            }
            if (i != max_i - 1) {
                read_block_height++;
            }
            if (j != max_j - 1) {
                read_block_width += channels;
            }

            cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index + dev_src_row_index, s_pitch, host_src + host_src_index, width * channels * sizeof(char), read_block_width * sizeof(char), read_block_height, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync1 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
            if (i == 0) {
                cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index, s_pitch, host_src + host_src_index, width * channels * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync2 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (i == max_i - 1) {
                if (i != 0) {
                    host_src_index += width * channels;
                }
                cudaStatus = cudaMemcpy2DAsync(dev_src + s_pitch * (block_height + 1) + dev_src_column_index, s_pitch, host_src + host_src_index + (block_height - 1) * width * channels, width * channels * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync3 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (j == 0) {
                for (int k = 0; k < block_height + 2; k++) {
                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + 125, dev_src + s_pitch * k + 128, channels, cudaMemcpyDeviceToDevice);
                    if (cudaStatus != cudaSuccess) {
                        cerr << "cudaMemcpyAsync4 failed!\n";
                        cudaFree(dev_src);
                        cudaFree(dev_dst);
                        return -1;
                    }
                }
            }
            if (j == max_j - 1) {
                for (int k = 0; k < block_height + 2; k++) {

                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + 128 + block_width * channels, dev_src + s_pitch * k + block_width * channels + 125, channels, cudaMemcpyDeviceToDevice);
                    if (cudaStatus != cudaSuccess) {
                        cerr << "cudaMemcpyAsync5 failed!\n";
                        cudaFree(dev_src);
                        cudaFree(dev_dst);
                        return -1;
                    }

                }
            }

            dim3 block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y);
            dim3 grid(((block_width + 3) / 4 + THREAD_PER_BLOCK_X - 1) / THREAD_PER_BLOCK_X, (block_height + THREAD_PER_BLOCK_Y - 1) / THREAD_PER_BLOCK_Y);
            
            KernelBlurFilterRGBOptimized << <grid, block >> > ((uint32_t*)dev_dst, (uint32_t*)dev_src, block_height, block_width, (int)s_pitch / sizeof(uint32_t), (int)d_pitch / sizeof(uint32_t));

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                cerr << "KernelBlurFilter launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }

            cudaStatus = cudaMemcpy2DAsync(host_dst + i * MATRIX_BLOCK_HEIGHT * width * channels + MATRIX_BLOCK_WIDTH * j * channels, width * channels * sizeof(char), dev_dst, d_pitch, block_width * channels * sizeof(char), block_height, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync6 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}
