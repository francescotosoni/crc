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

const int DIM = 1024;

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


int main() {

    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;

    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
   
    uint32_t hcrc = 0xffffffff;
    std::ifstream fin("../input3.txt");
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
    TM_host.print("CRC32C host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    //TM_device.start();

    uint32_t* dcrc;
    uint8_t* ddata;
    uint32_t* d_table;
    uint32_t *d_tmp;
    cudaMalloc(&ddata, length*sizeof(uint8_t));
    cudaMalloc(&d_table, length * 256 * sizeof(uint32_t));
    cudaMalloc(&d_tmp, length * sizeof(uint32_t));
    
    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVICE
    cudaMemcpy(ddata, data, length*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, table, length * 256 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    dim3 block_size(DIM, 1, 1);
    dim3 num_blocks(ceil((float)length/DIM), 1, 1);
   
    crc32kernel<<< num_blocks, block_size >>>(ddata, length, d_table, d_tmp);

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    uint32_t* h_crc_tmp = (uint32_t *)malloc(length * sizeof(uint32_t));
    cudaMemcpy(h_crc_tmp, d_tmp, length*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int i = 1; i < length; i++) {
        h_crc_tmp[0] ^= h_crc_tmp[i];
    }
    h_crc_tmp[0] ^= 0xffffffff;

    TM_device.stop();
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR
    TM_device.print("CRC32 device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // RESULT CHECK
    printf("0x%x - 0x%x\n", hcrc, h_crc_tmp[0]);
    
    if (hcrc != h_crc_tmp[0]) {
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
    cudaFree(dcrc);

    // -------------------------------------------------------------------------
    //cudaDeviceReset();
}
