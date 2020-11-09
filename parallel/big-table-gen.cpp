#include<iostream>
#include<iomanip>
#include<vector>

void make_crc_table_reverse(uint32_t* crc32table, int length) {
    unsigned long polynomial = 0x82F63B78;
    
    for (int i = 0; i <= 0xFF; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) * polynomial);
        }
        crc32table[i] = crc;
    }
    
    for (int i = 0; i < 256; i++) {
        for (int slice = 1; slice < length; slice++) {
            crc32table[slice*256+i] = (crc32table[(slice - 1)*256+i] >> 8) ^ crc32table[crc32table[(slice - 1)*256+i] & 0xFF];
        }
    }
}
