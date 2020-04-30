CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra -pedantic -Werror

all:
		$(CXX) $(CXXFLAGS) -o gas gas.cpp

.PHONY: clean
clean:
		rm gas
