all:
	g++ -std=c++14 test.cc -o test -march=native -O0 -g -Iasmjit/src/ -Lasmjit/build/ -lasmjit
fast:
	g++ -std=c++14 test.cc -o test -march=native -O3 -Iasmjit/src/ -Lasmjit/build/ -lasmjit
