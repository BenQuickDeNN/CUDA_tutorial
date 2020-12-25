rm main
nvcc -x -cu main.cpp -o main -O3 -Xcompiler="-std=c++11 -fopenmp"