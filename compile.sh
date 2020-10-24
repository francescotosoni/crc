#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/crc16-prl.cu -I"$DIR"/include -o crc16-prl
