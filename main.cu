#include "header.cuh"
#include "gray.cuh"
#include "rgb.cuh"

#define duration(a) chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() chrono::high_resolution_clock::now()
typedef chrono::high_resolution_clock::time_point TimeVar;

template<typename F, typename... Args>
double measureTime(F func, Args&&... args) {
    const TimeVar t1 = timeNow();
    func(forward<Args>(args)...);
    const TimeVar t2 = timeNow();
    return duration(t2 - t1) / 1000.;
}

void cmpResult(const unsigned char* a, const unsigned char* b, int size, const string& a_string, const string& b_string) {
    const int cmp_val = memcmp(a, b, size);
    cout << a_string << " ";
    cmp_val == 0 ? cout << "==" : cout << "!=";
    cout << " " << b_string << '\n';
}

void cpuBlurFilter(unsigned char* data, unsigned char* res, unsigned int w, unsigned int h, int c)
{
    for (int i = 0; i < h; i++) {

        unsigned char* cur_line = data + i * w * c;
        unsigned char* next_line = data + (i + 1) * w * c;
        unsigned char* prev_line = data + (i - 1) * w * c;
        unsigned char* res_line = res + i * w * c;
        for (int j = 0; j < w * c; j++) {
            int pixel = 0;
            if (i == 0 || i == h - 1) {
                if (i == 0) {
                    if (j - c < 0) {
                        pixel = 5 * cur_line[j] + 2 * cur_line[j + c] + 2 * next_line[j] + next_line[j + c];
                    }
                    if (j + c >= w * c) {
                        pixel = 2 * cur_line[j - c] + 5 * cur_line[j] + next_line[j - c] + 2 * next_line[j];
                    }
                    if (j - c >= 0 && j + c < w * c) {
                        pixel = 2 * cur_line[j - c] + 3 * cur_line[j] + 2 * cur_line[j + c] + next_line[j - c] + next_line[j] + next_line[j + c];
                    }
                }
                if (i == h - 1) {
                    if (j - c < 0) {
                        pixel = 5 * cur_line[j] + 2 * cur_line[j + c] + 2 * prev_line[j] + prev_line[j + c];
                    }
                    if (j + c >= w * c) {
                        pixel = 2 * cur_line[j - c] + 5 * cur_line[j] + prev_line[j - c] + 2 * prev_line[j];
                    }
                    if (j - c >= 0 && j + c < w * c) {
                        pixel = 2 * cur_line[j - c] + 3 * cur_line[j] + 2 * cur_line[j + c] + prev_line[j - c] + prev_line[j] + prev_line[j + c];
                    }
                }
            }
            else {
                if (j - c < 0) {
                    pixel = 2 * prev_line[j] + prev_line[j + c] + 3 * cur_line[j] + cur_line[j + c] + 2 * next_line[j] + next_line[j + c];
                }
                else {
                    if (j + c >= w * c) {
                        pixel = prev_line[j - c] + 2 * prev_line[j] + cur_line[j - c] + 3 * cur_line[j] + next_line[j - c] + 2 * next_line[j];
                    }
                    else {
                        pixel = prev_line[j - c] + prev_line[j] + prev_line[j + c] + cur_line[j - c] + 2 * cur_line[j] + cur_line[j + c] + next_line[j - c] + next_line[j] + next_line[j + c];
                    }
                }
            }
            pixel /= 10;
            res_line[j] = (unsigned char)pixel;
        }
    }
}

int main()
{
    string names[] = {"sea", "malpa"};

    for (int i = 0; i < (int)(sizeof(names) / sizeof(names[0]) - 1); i++) {

        unsigned char* data = nullptr;
        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int channels = 0;

        __loadPPM((names[i] + ".pgm").c_str(), &data, &width, &height, &channels);

        unsigned char* res_cpu = (unsigned char*)malloc(width * height * sizeof(unsigned char) * channels);
        memset(res_cpu, 0, width * height * sizeof(unsigned char) * channels);
                       
        const double cpu_time = measureTime(cpuBlurFilter, data, res_cpu, width, height, channels);
        __savePPM((names[i] + "_cpu.pgm").c_str(), res_cpu, width, height, channels);
        cout << "\nCPU: " + names[i] + " " << width << 'x' << height << " - " << cpu_time << " ms\n";

        unsigned char* res_gpu = (unsigned char*)malloc(width * height * sizeof(unsigned char) * channels);

        memset(res_gpu, 0, width * height * sizeof(unsigned char) * channels);
        double gpu_time = gpuBlurFilterGray(res_gpu, data, width, height);
        cout << "GPU: " + names[i] + " " << width << 'x' << height << " - " << gpu_time << " ms\n";
        cmpResult(res_cpu, res_gpu, width * height * channels * sizeof(unsigned char), "CPU", "GPU");
        __savePPM((names[i] + "_gpu.pgm").c_str(), res_gpu, width, height, channels);

        memset(res_gpu, 0, width * height * sizeof(unsigned char) * channels);
        double gpu_opt_time = gpuBlurFilterGrayOptimized(res_gpu, data, width, height);
        cout << "GPU optimized: " + names[i] + " " << width << 'x' << height << " - " << gpu_opt_time << " ms\n";
        cmpResult(res_cpu, res_gpu, width * height * channels * sizeof(unsigned char), "CPU", "GPU optimized");
        __savePPM((names[i] + "_gpu2.pgm").c_str(), res_gpu, width, height, channels);

        free(data);
        free(res_cpu);
        free(res_gpu);
    }

    for (int i = 1; i < sizeof(names) / sizeof(names[0]); i++) {

        unsigned char* data = nullptr;
        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int channels = 0;

        __loadPPM((names[i] + ".ppm").c_str(), &data, &width, &height, &channels);
        unsigned char* res_cpu = (unsigned char*)malloc(width * height * sizeof(unsigned char) * channels);
        memset(res_cpu, 0, width * height * sizeof(unsigned char) * channels);
                
        const double cpu_time = measureTime(cpuBlurFilter, data, res_cpu, width, height, channels);
        __savePPM((names[i] + "_cpu.ppm").c_str(), res_cpu, width, height, channels);
        cout << "\nCPU: " + names[i] + " " << width << 'x' << height << " - " << cpu_time << " ms\n";
        
        unsigned char* res_gpu = (unsigned char*)malloc(width * height * sizeof(unsigned char) * channels);

        memset(res_gpu, 0, width * height * sizeof(unsigned char) * channels);
        double gpu_time = gpuBlurFilterRGB(res_gpu, data, width, height);
        cout << "GPU: " + names[i] + " " << width << 'x' << height << " - " << gpu_time << " ms\n";
        __savePPM((names[i] + "_gpu.ppm").c_str(), res_gpu, width, height, channels);
        cmpResult(res_cpu, res_gpu, width * height * channels * sizeof(unsigned char), "CPU", "GPU");
        
        memset(res_gpu, 0, width * height * sizeof(unsigned char) * channels);
        double gpu_opt_time = gpuBlurFilterRGBOptimized(res_gpu, data, width, height);
        cout << "GPU optimized: " + names[i] + " " << width << 'x' << height << " - " << gpu_opt_time << " ms\n";
        __savePPM((names[i] + "_gpu2.ppm").c_str(), res_gpu, width, height, channels);
        cmpResult(res_cpu, res_gpu, width * height * channels * sizeof(unsigned char), "CPU", "GPU optimized");
        
        free(data);
        free(res_cpu);
        free(res_gpu);
    }

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceReset failed!\n";
        return 1;
    }

    /*for (int i = 0; i < height; i++) {
            for (int j = 0; j < width * channels; j++) {
                if (res_cpu[i * width * channels + j] != res_gpu[i * width * channels + j]) {
                    cout << (unsigned int)res_cpu[i * width * channels + j] << ' ' << (unsigned int)res_gpu[i * width * channels + j] << ' ' << i << ' ' << j << '\n';
                }
            }
        }*/
    return 0;
}

