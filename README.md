# PARALLEL CRC CALCULATION
A software implementation of crc calculation in different variants. This work includes a classic version of the Sarwate CRC32 algorithm and a parallel variant of it, made with Cuda. In particular, a comparison between the performance of serial and some parallel approaches has been realized.

## Repo content
1. Serial implementation (bytewise):
    - [CRC8](./serial/crc8.cpp)
    - [CRC16_CCITT_FALSE](./serial/crc8.cpp)
    - [CRC32_C](./serial/crc32.cpp)
    - [CRC64_WE](./serial/crc64.cpp)
2. Parallel implementation: 
    - CRC32 with final XOR done by host
    - CRC32 with final XOR done by reduction in device (v1)
    - CRC32 with final XOR done by reduction in device (v2)

