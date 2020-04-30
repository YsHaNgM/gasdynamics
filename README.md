`gas.cpp` contains a simple 1D Lagrangian gas dynamics code.

Please complete the following tasks:

1. There are some syntactic errors in the code that cause it to fail to compile,
   please fix these.
2. Convert the provided Makefile to modern CMake (https://cmake.org/);
3. Implement MPI inter-node parallelisation;
4. Implement OpenMP intra-node parallelisation (MPI+OpenMP);
5. Explore performance with varying process/thread counts and varying problem
   sizes (within the bounds of compute available to you, if that is only e.g.
   two or four CPU cores that is not a problem).

The source contains several comments marked XXX giving hints about how to
proceed with the parallelisation.

The provided `plot()` function can be called to print CSV output to standard
out. These files can be compared directly (e.g. with `diff`), or plotted with
any chart generator (e.g. PyPlot or Excel). Two `.png` files are included showing the
density field for 100 cell and 100,000 cell runs.  Any changes should not
perceptibly alter the output for a fixed number of cells (very minor deviations
of the order of floating-point round-off may occur depending on compiler, flags
and system).

This exercise is expected to take 2-3 hours.
