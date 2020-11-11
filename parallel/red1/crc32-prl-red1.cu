#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include "tables.hpp"
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

const int DIM = 128;

__global__
void crc32kernel(uint8_t* data, int length, uint32_t* d_table, uint32_t* tmp) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t crc = 0xffffffff;
    uint32_t* current = (uint32_t*)data;
    
    if(id < 4) { 
        uint32_t one = *current ^ crc;
        int i = (256*((length-1)-id))+((one>>(8*id)) & 0xff);
        tmp[id] = d_table[i];
    }
    else if(id > 3 && id < length) {
        uint32_t two = *(current+(id / 4));
        int i = (256*((length-1)-id))+(two>>(8*(id % 4)) & 0xff);
        tmp[id] = d_table[i];
    }

}

__global__
void xorkernel(uint32_t* d_input) {
    __shared__ uint32_t s_mem[DIM];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    s_mem[tx] = d_input[id];
    __syncthreads();

    for(int i = blockDim.x/2; i > 0; i >>= 1){
        if(tx < i){
            s_mem[tx] ^= s_mem[tx + i];
        }
        __syncthreads();
        
    }

    if(tx == 0){
        d_input[blockIdx.x] = s_mem[0];
    }
}


int main() {

    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
   
    uint32_t hcrc = 0xffffffff;
    std::ifstream fin("../input1.txt");
    std::string temp;
    std::string d("");

    if(fin.is_open()){
        while(getline(fin, temp)){
            d.append(temp);
        }
    }
    fin.close();

    std::cout << d.length() << std::endl;
    uint8_t data[d.length()];

    for(int i=0; i<d.length(); i++){
        data[i] = d[i];
    }

    int length = sizeof(data);
    auto *table = (uint32_t *)malloc(length * 256 * sizeof(uint32_t));
    make_crc_table_reverse(table, length);
    
    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for(int i = 0; i < length; i++) {
        hcrc = table[(hcrc ^ data[i]) & 0xFF] ^ (hcrc>>8);
    }

    hcrc ^= 0xffffffff;

    TM_host.stop();
    TM_host.print("CRC32 host:   ");
    
    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    //TM_device.start();

    uint8_t* ddata;
    uint32_t* d_table;
    uint32_t *tmp;
    cudaMalloc(&ddata, length*sizeof(uint8_t));
    cudaMalloc(&d_table, length * 256 * sizeof(uint32_t));
    cudaMalloc(&tmp, length * sizeof(uint32_t));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVICE
    cudaMemcpy(ddata, data, length*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, table, length * 256 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    int n_temp = length;
    int N_iter = 0;
    
    while(n_temp > 1){
        n_temp/=DIM;
        N_iter++;
    }   
   
    int tile_temp = DIM;

    TM_device.start();

    dim3 block_size(1024, 1, 1);
    dim3 num_blocks(ceil((float)length/1024), 1, 1);
   
    crc32kernel<<< num_blocks, block_size >>>(ddata, length, d_table, tmp);
    
    for(int i = 0; i < N_iter; i++){
        xorkernel<<<ceil((float)length/tile_temp), DIM>>>(tmp);
        tile_temp *= DIM;
    }
    TM_device.stop();

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    cudaDeviceSynchronize();
    uint32_t h_crc_tmp;
    cudaMemcpy(&h_crc_tmp, tmp, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    h_crc_tmp ^= 0xffffffff;

    CHECK_CUDA_ERROR
    TM_device.print("CRC32 device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // RESULT CHECK
    printf("0x%x - 0x%x\n", hcrc, h_crc_tmp);
    
    if (hcrc != h_crc_tmp) {
        std::cout << "CRC value mismatch\n\n";
        //cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    //delete[] hdata;
    free(table);

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    cudaFree(d_table);
    cudaFree(ddata);

    // -------------------------------------------------------------------------
    //cudaDeviceReset();
}
