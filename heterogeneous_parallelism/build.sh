CC="nvcc"
INC="include"
SRC1="source/matmul.cpp"
SRC2="source/matmul_large.cpp"
BIN1="bin/matmul.exe"
BIN2="bin/matmul_large"
OPT=""

rm $BIN1
rm $BIN2

$CC -x cu $SRC1 -o $BIN1 -I $INC $OPT
$CC -x cu $SRC2 -o $BIN2 -I $INC $OPT