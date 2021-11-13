#include <cudnn.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#define cudaCheckError(status) { cudaAssert(status, __FILE__, __LINE__); }
inline auto cudaAssert(cudaError_t status, const char* file, int line) -> void {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA API error \"%s\" at %s:%i\n", cudaGetErrorString(status), file, line);
        exit(EXIT_FAILURE);
    }
}

#define cudnnCheckError(status) { cudnnAssert(status, __FILE__, __LINE__); }
inline auto cudnnAssert(cudnnStatus_t status, const char* file, int line) -> void {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN API error \"%s\" at %s:%i\n", cudnnGetErrorString(status), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_1D_KERNEL_LOOP(idx, n)                                 \
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (n); \
       idx += blockDim.x * gridDim.x)

__global__
void float2half(const float* input, std::size_t size, half* output) {
    CUDA_1D_KERNEL_LOOP(idx, size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__
void half2float(const half* input, std::size_t size, float* output) {
    CUDA_1D_KERNEL_LOOP(idx, size) {
        output[idx] = __half2float(input[idx]);
    }
}

auto main(int argc, const char** argv) -> int {
    if (argc != 2) {
        std::cout << "usage: conv2d <filename>\n";
        return -1;
    }
    
    const char* filename = argv[1];
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        throw std::runtime_error("cv::imread() failed: image not found");
    }
    
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    const std::size_t image_size = image.total() * image.channels();
    
    cudnnHandle_t cudnn_handle;
    cudnnCheckError(cudnnCreate(&cudnn_handle));

    float* d_input = nullptr;
    cudaCheckError(cudaMalloc(&d_input, image_size * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_input, image.ptr<float>(0), image_size * sizeof(float), cudaMemcpyDefault));
    
    half* fp16_image = nullptr;
    cudaCheckError(cudaMalloc(&fp16_image, image_size * sizeof(half)));

    float2half<<<1, 64>>>(d_input, image_size, fp16_image);
    cudaFree(d_input);

    cudnnTensorDescriptor_t input_desc;
    cudnnCheckError(cudnnCreateTensorDescriptor(&input_desc));
    cudnnCheckError(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1,
        image.channels(),
        image.rows,
        image.cols
    ));

    cudnnFilterDescriptor_t filter_desc;
    cudnnCheckError(cudnnCreateFilterDescriptor(&filter_desc));
    cudnnCheckError(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_HALF,
        CUDNN_TENSOR_NCHW,
        image.channels(),
        image.channels(),
        3,
        3
    ));

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCheckError(cudnnCreateConvolutionDescriptor(&conv_desc));
    cudnnCheckError(cudnnSetConvolution2dDescriptor(
        conv_desc,
        1, 1,
        1, 1,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_HALF
    ));

    cudnnCheckError(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    cudnnTensorDescriptor_t output_desc;
    cudnnCheckError(cudnnCreateTensorDescriptor(&output_desc));
    cudnnCheckError(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        1,
        image.channels(),
        image.rows,
        image.cols
    ));

    cudnnConvolutionFwdAlgo_t fwd_algo;
    int requested_algo_count;
    int algo_count;

    cudnnCheckError(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &requested_algo_count));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
    cudnnCheckError(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &algo_count,
        perf_results.data()
    ));

    fwd_algo = perf_results.front().algo;

    std::size_t workspace_size = 0;
    cudnnCheckError(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        fwd_algo,
        &workspace_size
    ));

    void* d_workspace = nullptr;
    cudaCheckError(cudaMalloc(&d_workspace, workspace_size));

    const std::vector<half> filter = {
        0.0625, 0.125, 0.0625,
        0.125, 0.25, 0.125,
        0.0625, 0.125, 0.0625
    };

    std::vector<half> h_filter;
    for (std::size_t idx = 0; idx < 3 * 3; ++idx) {
        for (const auto& val : filter) {
            h_filter.emplace_back(val);
        }
    }
    
    half* d_filter = nullptr;
    const std::size_t filter_size = h_filter.size();
    cudaCheckError(cudaMalloc(&d_filter, filter_size * sizeof(half)));
    cudaCheckError(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(half), cudaMemcpyDefault));
    
    half* d_output = nullptr;
    cudaCheckError(cudaMalloc(&d_output, image_size * sizeof(half)));

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudnnCheckError(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc,
        fp16_image,
        filter_desc,
        d_filter,
        conv_desc,
        fwd_algo,
        d_workspace,
        workspace_size,
        &beta,
        output_desc,
        d_output
    ));

    float* fp32_output = nullptr;
    cudaCheckError(cudaMalloc(&fp32_output, image_size * sizeof(float)));

    half2float<<<1, 64>>>(d_output, image_size, fp32_output);
    cudaFree(d_output);
    
    cv::Mat output(image.rows, image.cols, CV_32FC3);
    cudaCheckError(cudaMemcpy(output.ptr<float>(0), fp32_output, image_size * sizeof(float), cudaMemcpyDefault));

    cv::normalize(output, output, 0.0, 255.0, cv::NORM_MINMAX);
    output.convertTo(output, CV_8UC3);

    cv::imshow("output", output);
    cv::waitKey();

    cv::imwrite("output.png", output);

    cudaFree(d_filter);
    cudaFree(d_workspace);
    cudaFree(fp16_image);
    cudaFree(fp32_output);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(output_desc);

    cudnnDestroy(cudnn_handle);

    return 0;
}
