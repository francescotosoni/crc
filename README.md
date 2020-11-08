# PARALLEL CRC CALCULATION
A software implementation of crc calculation in different variants. This work includes a classic version of the Sarwate CRC32 algorithm and a parallel variant of it, made with Cuda. In particular, a comparison between the performance of serial and some parallel approaches has been realized.



## Repo content
1. Serial implementation (bytewise):
    - [CRC8](./serial/crc8.cpp)
    - [CRC16_CCITT_FALSE](./serial/crc16.cpp)
    - [CRC32_C](./serial/crc32.cpp)
    - [CRC64_WE](./serial/crc64.cpp)
2. Parallel implementation: 
    - CRC32 with final XOR done by [host](./parallel/host/crc32-prl-host.cu)
    - CRC32 with final XOR done by reduction in device ([v1](./parallel/host/crc32-prl-red1.cu))
    - CRC32 with final XOR done by reduction in device ([v2](./parallel/host/crc32-prl-red2.cu))
3. Table [generator](./parallel/big-table-gen.cpp) for the parallel approach

The main differences between the parallel versions, is the realization of the final *xor* operation of the "mid-CRCs" calculated in the first kernel method (the same for all three variants). It is first executed sequentially by the host; in the other two, it is executed in devices using two similar reduction techniques  in a specific kernel method. The best performance is obtained from v2.

**NOTE**: In the serial section there are also the 8, 16 and 64 bit CRC versions as an *example*, since the algorithm is practically the same. Only the 32 bit version is present in both serial and parallel versions. 

In particular, the CRC32-C version is realized, i.e. the polynomial `0x1EDC6F41` is used to calculate the checksum value. If you prefer to have the traditional CRC32 (polynomial `0x4C11DB7`) it is enough to replace it in the generation of the tables. 
In addition, three input files are provided as tests in parallel directory. 

## Usage
Each file has its own main, so it must be compiled and run separately. For the serial variant, simply compile the source file with `g++` and run it. For the parallel, just navigate to the subdirectory of the version to execute and write the following command: 
```
bash compile.sh && ./crc32-prl
```
**NOTE**: Serial programs have already the table itself, while for parallel programs it is dynamically generated. 
