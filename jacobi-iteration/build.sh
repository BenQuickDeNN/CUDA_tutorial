CC="nvcc"
SRC="source/jacobi-1d.cu"
BIN="bin/jacobi-1d"
INC="include"
FLAG=""

rm -f $BIN

$CC $SRC -o $BIN -I $INC $FLAG