`gas.cpp` contains a simple 1D Lagrangian gas dynamics code.

The provided `plot()` function can be called to print CSV output to standard
out. These files can be compared directly (e.g. with `diff`), or plotted with
any chart generator (e.g. PyPlot or Excel).

Tested with CMake 3.13 & 3.16 use both gcc9, clang9 and apple clang (failed on C++17 
shared_ptr).

omp off:

mpirun -n 2 ./gas 100000 takes 308826.750668 ms

mpirun -n 1 ./gas 100000 takes 393744.256646 ms

omp on:

mpirun -n 2 ./gas 100000 takes 356381.796035 ms

mpirun -n 1 ./gas 100000 takes 448773.837776 ms


Using OpenMP seems not benefit in such code, maybe too much data race in reading/writing 
of same array, which implicitly causes lock. Iteration also rely on previous result.
By changing omp parameters from shared into private, preformance improved.
