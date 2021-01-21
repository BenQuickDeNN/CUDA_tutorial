INC="include"

rm main
nvcc -x cu -g main.cu -o main -O3 -I $INC -Xcompiler="-std=c++11 -fopenmp"