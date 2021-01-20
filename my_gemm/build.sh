INC="include"

rm main
nvcc -x cu main.cpp -o main -O3 -I $INC -Xcompiler="-std=c++11"