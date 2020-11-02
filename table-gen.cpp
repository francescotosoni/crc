#include <iostream>
#include <iomanip>

void make_crc_table_reverse(unsigned long crcTable[]) {
    unsigned long POLYNOMIAL = 0x82F63B78;
    unsigned long remainder;
    unsigned char b = 0;
    do {
        // Start with the data byte
        remainder = b;
        for (unsigned long bit = 8; bit > 0; --bit) {
            if (remainder & 1)
                remainder = (remainder >> 1) ^ POLYNOMIAL;
            else
                remainder = (remainder >> 1);
        }
        crcTable[(size_t)b] = remainder;
    } while(0 != ++b);
}

void make_crc_table_forward(unsigned long crcTable[]) {
    const uint polynomial = 0x1EDC6F41;
    //crcTable = new uint[256];

    for (int divident = 0; divident < 256; divident++) // iterate over all possible input byte values 0 - 255 
    {
        uint curByte = (uint)(divident << 24); // move divident byte into MSB of 32Bit CRC 
        for (unsigned long bit = 0; bit < 8; bit++)
        {
            if ((curByte & 0x80000000) != 0)
            {
                curByte <<= 1;
                curByte ^= polynomial;
            }
            else
            {
                curByte <<= 1;
            }
        }

        crcTable[divident] = curByte;
    }
} 

int main() {
    unsigned long crcTable[256];
    make_crc_table_reverse(crcTable);
    // Print the CRC table
    for (size_t i = 0; i < 256; i++) {
        std::cout << std::setfill('0') << std::setw(8) << std::hex << crcTable[i];
        if (i % 4 == 3)
            std::cout << std::endl;
        else
            std::cout << ", ";
    }
    return 0;
}