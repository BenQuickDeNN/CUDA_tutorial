CC="nvcc"
SRC="source/vadd.cu"
BIN="bin/vadd"
INC="include"
OPT=""

$CC -x cu $SRC -o $BIN -I $INC $OPT