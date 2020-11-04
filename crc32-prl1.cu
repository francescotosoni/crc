#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "tables.hpp"
/*#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;*/

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__
void crc32kernel(uint8_t* data, uint32_t* crc, int length, uint32_t* d_table, uint32_t* tmp) {
    /// YOUR CODE
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    *crc = 0xffffffff;
    uint32_t* current = (uint32_t*)data;
    
    if(id < 4) { 
        uint32_t one = *current ^ *crc;
        int i = (256*(15-id))+((one>>(8*id)) & 0xff);
        tmp[id] = d_table[i];
        crc[0] = tmp[0];
    }
    else if(id > 3 && id < length) {
        uint32_t two = *(current+(id / 4));
        int i = (256*(15-id))+(two>>(8*(id % 4)) & 0xff);
        tmp[id] = d_table[i];
    }

    __syncthreads();

    if(id == 16) {
        for(int i = 1; i < length; i++) {
           crc[0] ^= (tmp[i]);
        }
        crc[0] ^= 0xffffffff;
    }
}

int main() {
    //Timer<DEVICE> TM_device;
    //Timer<HOST>   TM_host;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
   
    uint32_t hcrc = 0xffffffff;
    uint8_t data[] = "ciaobelocomestaisanhasbj";
    int length = sizeof(data) - 1;

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    
    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    //TM_host.start();

    for(int i = 0; i < length; i++) {
        hcrc = table_s[(hcrc ^ data[i]) & 0xFFL] ^ (hcrc>>8);
    }

    hcrc ^= 0xffffffff;

    //TM_host.stop();
    //TM_host.print("CRC32C host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    //TM_device.start();

    uint32_t crc = 0xffffffff;
    //uint8_t data[] = "ciaobelocomestai";

    uint32_t* dcrc;
    uint8_t* ddata;
    uint32_t* d_table;
    uint32_t *tmp;
    cudaMalloc(&dcrc, length*sizeof(uint32_t));
    cudaMalloc(&ddata, length*sizeof(uint8_t));
    cudaMalloc(&d_table, 16 * 256 * sizeof(uint32_t));
    cudaMalloc(&tmp, length * sizeof(uint32_t));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVICE
    //SAFE_CALL(cudaMemcpy(dcrc, crc, sizeof(uint16_t), cudaMemcpyHostToDevice));
    cudaMemcpy(ddata, data, length*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, table_m, 16 * 256 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    //TM_device.start();

    dim3 block_size(256, 1, 1);
    dim3 num_blocks(ceil((float)N/256), 1, 1);
   
    crc32kernel<<< num_blocks, block_size >>>(ddata, dcrc, length, d_table, tmp);

    /*TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("CRC32C device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";*/

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    cudaDeviceSynchronize();
    uint32_t h_crc_tmp;
    cudaMemcpy(&h_crc_tmp, dcrc, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    //TM_device.stop();
    //CHECK_CUDA_ERROR
    //TM_device.print("CRC32 device: ");

   /* std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";*/

    // -------------------------------------------------------------------------
    // RESULT CHECK
    printf("0x%x - 0x%x\n", hcrc, h_crc_tmp);
    
    if (hcrc != h_crc_tmp) {
        /*std::cerr << "wrong result at: ("
                << (i / N) << ", " << (i % N) << ")"
                << "\nhost:   " << h_matrixC[i]
                << "\ndevice: " << h_matrix_tmp[i] << "\n\n";*/
        //std::cerr << "wrong result\n\n";
        //cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    //delete[] hdata;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    cudaFree(d_table);
    cudaFree(ddata);
    cudaFree(dcrc);

    // -------------------------------------------------------------------------
    //cudaDeviceReset();
}
