rm main
nvcc -x cu main.cu -o main -O3 -Xcompiler="-std=c++11 -fopenmp"