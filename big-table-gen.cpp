#include<iostream>
#include<iomanip>

void make_crc_table_reverse(uint32_t crc32table[16][256]) {
    unsigned long polynomial = 0x82F63B78;
    
    for (int i = 0; i <= 0xFF; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) * polynomial);
        }
        crc32table[0][i] = crc;
    }
    
    for (int i = 0; i < 256; i++) {
        for (int slice = 1; slice < 16; slice++) {
        crc32table[slice][i] = (crc32table[slice - 1][i] >> 8) ^ crc32table[0][crc32table[slice - 1][i] & 0xFF];
        }
    }
}

    int main() {
    uint32_t crc32table[16][256];
    make_crc_table_reverse(crc32table);
    // Print the CRC table
    for (int j = 0; j < 16; j++) {
        for (size_t i = 0; i < 256; i++) {
            std::cout <<"0x" << std::setfill('0') << std::setw(8) <<  std::hex << crc32table[j][i];
            if (i % 4 == 3)
                std::cout << ", " << std::endl;
            else
                std::cout << ", ";
        }
        std::cout << std::endl << std::endl << std::endl;
    }
    return 0;
}
