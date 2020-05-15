CC2="nvcc"
SRC2="source/cuda_vadd.cpp"
BIN2="bin/cuda_vadd"
INC="include"
OPT2=""

rm $BIN2

$CC2 -x cu $SRC2 -o $BIN2 -I $INC $OPT2