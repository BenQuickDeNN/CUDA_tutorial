INC="include"

rm sample_mm
nvcc -x cu sample_mm.cu -o sample_mm -O3 -I $INC -Xcompiler="-std=c++11"