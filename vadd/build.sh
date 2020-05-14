CC="nvcc"
SRC="source/vadd.cpp"
BIN="bin/vadd"
INC="include"
OPT=""

rm $BIN

$CC -x cu $SRC -o $BIN -I $INC $OPT