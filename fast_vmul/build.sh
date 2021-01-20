rm main
nvcc -x cu main.cu -o main -O3 -I $INC -Xcompiler="-std=c++11"