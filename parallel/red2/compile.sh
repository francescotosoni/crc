#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/crc32-prl-red2.cu "../$DIR"/big-table-gen.cpp -I"../$DIR"/include -o crc32-prl