add_executable(gas gas.cpp)


target_link_libraries(gas PUBLIC OpenMP::OpenMP_CXX ${ALL_THIRD_PARTY_LIBRARIES})