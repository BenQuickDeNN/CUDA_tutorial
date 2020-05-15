CC1="g++"
CC2="nvcc"
SRC1="source/cpu_vadd.cpp"
SRC2="source/cuda_vadd.cpp"
BIN1="bin/cpu_vadd"
BIN2="bin/cuda_vadd"
INC="include"
OPT1="-O3 -fopenmp -Wall"
OPT2=""

rm $BIN1
rm $BIN2

$CC1 $SRC1 -o $BIN1 -I $INC $OPT1
$CC2 -x cu $SRC2 -o $BIN2 -I $INC $OPT2