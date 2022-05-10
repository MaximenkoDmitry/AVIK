#include "header.cuh"

__global__ void KernelBlurFilter(unsigned char* dst, unsigned char* src, int height, int width, int s_pitch, int d_pitch) {
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_center = d_pitch * thread_y + thread_x;
    int src_center = s_pitch * (thread_y + 1) + thread_x + 1;
    int src_low = s_pitch * thread_y + thread_x + 1;
    int src_high = s_pitch * (thread_y + 2) + thread_x + 1;
    int pixel = 0;
    if (thread_x < width && thread_y < height) {
        pixel = src[src_high - 1] + src[src_high] + src[src_high + 1];
        pixel += src[src_center - 1] + 2 * src[src_center] + src[src_center + 1];
        pixel += src[src_low - 1] + src[src_low] + src[src_low + 1];
        pixel /= 10;
        dst[dst_center] = (unsigned char)pixel;
    }
}

__global__ void KernelBlurFilterOptimized(uint32_t* dst, uint32_t* src, int height, int width, int s_pitch, int d_pitch) {
    __shared__ unsigned char mem[(4 * THREAD_PER_BLOCK_X + 8) * (THREAD_PER_BLOCK_Y + 2)];  

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_center = d_pitch * thread_y + thread_x;
    int src_center = s_pitch * (thread_y + 1) + thread_x + 128 / 4; 
    int src_low = s_pitch * thread_y + thread_x + 128 / 4;
    int src_high = s_pitch * (thread_y + 2) + thread_x + 128 / 4;

    int mem_center = (threadIdx.y + 1) * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 4;  
    int mem_low = threadIdx.y * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 4;
    int mem_high = (threadIdx.y + 2) * (4 * THREAD_PER_BLOCK_X + 8) + threadIdx.x * 4 + 4;

    int mem32_center = mem_center / 4;
    int mem32_low = mem_low / 4;
    int mem32_high = mem_high / 4;

    uint32_t* mem32 = (uint32_t*)mem;

    if (thread_x * 4 <= width && thread_y <= height) {  
        mem32[mem_center / 4] = src[src_center]; 

        __syncthreads(); //курлык

        if (threadIdx.x == 0) {
            mem32[mem32_center - 1] = src[src_center - 1];
            mem32[mem32_high - 1] = src[src_high - 1];
            mem32[mem32_low - 1] = src[src_low - 1];
        }
        if (threadIdx.x == blockDim.x - 1) {
            mem32[mem32_center + 1] = src[src_center + 1];
            mem32[mem32_high + 1] = src[src_high + 1];
            mem32[mem32_low + 1] = src[src_low + 1];
        }
       
        if (threadIdx.y == 0) {
            mem32[mem32_low] = src[src_low];
        }
        if (threadIdx.y == blockDim.y - 1) {
            mem32[mem32_high] = src[src_high];
        }
        
        __syncthreads();

        uint32_t res = 0;
        for (int i = 0; i < 4; i++) {
            int32_t pixel = 0;
            pixel = mem[mem_high - 1 + i] + mem[mem_high + i] + mem[mem_high + 1 + i];
            pixel += mem[mem_center - 1 + i] + 2 * mem[mem_center + i] + mem[mem_center + 1 + i];
            pixel += mem[mem_low - 1 + i] + mem[mem_low + i] + mem[mem_low + 1 + i];
            pixel /= 10;
            res |= (pixel << (8 * i));
        }
        dst[dst_center] = res;
    }
}

double gpuBlurFilterGray(unsigned char* host_dst, unsigned char* host_src, int width, int height) {
    cudaError_t cudaStatus = cudaSuccess;
    unsigned char* dev_src = nullptr;
    unsigned char* dev_dst = nullptr;
    size_t s_pitch = 0;
    size_t d_pitch = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaMallocPitch((void**)&dev_src, &s_pitch, (2 + MATRIX_BLOCK_WIDTH) * sizeof(unsigned char), 2 + MATRIX_BLOCK_HEIGHT);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocPitch failed!\n";
        return -1;
    }

    cudaStatus = cudaMallocPitch((void**)&dev_dst, &d_pitch, MATRIX_BLOCK_WIDTH * sizeof(unsigned char), MATRIX_BLOCK_HEIGHT);
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
            int host_src_index = i * MATRIX_BLOCK_HEIGHT * width + MATRIX_BLOCK_WIDTH * j;
            int dev_src_column_index = 1;
            int dev_src_row_index = (int)s_pitch;
            int read_block_width = block_width;
            int read_block_height = block_height;
            if (j != 0) {
                dev_src_column_index--;
                host_src_index--;
                read_block_width++;
            }
            if (i != 0) {
                dev_src_row_index -= (int)s_pitch;
                host_src_index -= width;
                read_block_height++;
            }
            if (i != max_i - 1) {
                read_block_height++;
            }
            if (j != max_j - 1) {
                read_block_width++;
            }

            cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index + dev_src_row_index, s_pitch, host_src + host_src_index, width * sizeof(char), read_block_width * sizeof(char), read_block_height, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync1 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
            if (i == 0) {
                cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index, s_pitch, host_src + host_src_index, width * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync2 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (i == max_i - 1) {
                if (i != 0) {
                    host_src_index += width;
                }
                cudaStatus = cudaMemcpy2DAsync(dev_src + s_pitch * (block_height + 1) + dev_src_column_index, s_pitch, host_src + host_src_index + (block_height - 1) * width, width * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync3 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (j == 0) {
                for (int k = 0; k < block_height + 2; k++) {
                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k, dev_src + s_pitch * k + 1, 1, cudaMemcpyDeviceToDevice);
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
                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + 1 + block_width, dev_src + s_pitch * k + block_width, 1, cudaMemcpyDeviceToDevice);
                    if (cudaStatus != cudaSuccess) {
                        cerr << "cudaMemcpyAsync5 failed!\n";
                        cudaFree(dev_src);
                        cudaFree(dev_dst);
                        return -1;
                    }
                }
            }

            dim3 block(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y);
            dim3 grid((block_width + THREAD_PER_BLOCK_X - 1) / THREAD_PER_BLOCK_X, (block_height + THREAD_PER_BLOCK_Y - 1) / THREAD_PER_BLOCK_Y);
            
            KernelBlurFilter << <grid, block >> > (dev_dst, dev_src, block_height, block_width, (int)s_pitch, (int)d_pitch);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                cerr << "KernelBlurFilter launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }

            cudaStatus = cudaMemcpy2DAsync(host_dst + i * MATRIX_BLOCK_HEIGHT * width + MATRIX_BLOCK_WIDTH * j, width * sizeof(char), dev_dst, d_pitch, block_width * sizeof(char), block_height, cudaMemcpyDeviceToHost);
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

double gpuBlurFilterGrayOptimized(unsigned char* host_dst, unsigned char* host_src, int width, int height) {
    cudaError_t cudaStatus = cudaSuccess;
    unsigned char* dev_src = nullptr;
    unsigned char* dev_dst = nullptr;
    size_t s_pitch = 0;
    size_t d_pitch = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaMallocPitch((void**)&dev_src, &s_pitch, (2 + MATRIX_BLOCK_WIDTH) * sizeof(unsigned char), 2 + MATRIX_BLOCK_HEIGHT); 
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocPitch failed!\n";
        return -1;
    }

    cudaStatus = cudaMallocPitch((void**)&dev_dst, &d_pitch, MATRIX_BLOCK_WIDTH * sizeof(unsigned char), MATRIX_BLOCK_HEIGHT);
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
            int host_src_index = i * MATRIX_BLOCK_HEIGHT * width + MATRIX_BLOCK_WIDTH * j;
            int dev_src_column_index = 128;
            int dev_src_row_index = (int)s_pitch;
            int read_block_width = block_width;
            int read_block_height = block_height;
            if (j != 0) {
                dev_src_column_index--;
                host_src_index--;
                read_block_width++;
            }
            if (i != 0) {
                dev_src_row_index -= (int)s_pitch;
                host_src_index -= width;
                read_block_height++;
            }
            if (i != max_i - 1) {
                read_block_height++;
            }
            if (j != max_j - 1) {
                read_block_width++;
            }

            cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index + dev_src_row_index, s_pitch, host_src + host_src_index, width * sizeof(char), read_block_width * sizeof(char), read_block_height, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                cerr << "cudaMemcpy2DAsync1 failed!\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }
            if (i == 0) {
                cudaStatus = cudaMemcpy2DAsync(dev_src + dev_src_column_index, s_pitch, host_src + host_src_index, width * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync2 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (i == max_i - 1) {
                if (i != 0) {
                    host_src_index += width;
                }
                cudaStatus = cudaMemcpy2DAsync(dev_src + s_pitch * (block_height + 1) + dev_src_column_index, s_pitch, host_src + host_src_index + (block_height - 1) * width, width * sizeof(char), read_block_width * sizeof(char), 1, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    cerr << "cudaMemcpy2DAsync3 failed!\n";
                    cudaFree(dev_src);
                    cudaFree(dev_dst);
                    return -1;
                }
            }
            if (j == 0) {
                for (int k = 0; k < block_height + 2; k++) {
                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + 127, dev_src + s_pitch * k + 128, 1, cudaMemcpyDeviceToDevice);
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

                    cudaStatus = cudaMemcpyAsync(dev_src + s_pitch * k + 128 + block_width, dev_src + s_pitch * k + block_width + 127, 1, cudaMemcpyDeviceToDevice);
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
            
            KernelBlurFilterOptimized << <grid, block >> > ((uint32_t*)dev_dst, (uint32_t*)dev_src, block_height, block_width, (int)s_pitch / sizeof(uint32_t), (int)d_pitch / sizeof(uint32_t));

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                cerr << "KernelBlurFilter launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
                cudaFree(dev_src);
                cudaFree(dev_dst);
                return -1;
            }

            cudaStatus = cudaMemcpy2DAsync(host_dst + i * MATRIX_BLOCK_HEIGHT * width + MATRIX_BLOCK_WIDTH * j, width * sizeof(char), dev_dst, d_pitch, block_width * sizeof(char), block_height, cudaMemcpyDeviceToHost);
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
