#include <mitsuba/core/platform.h>

#include <iostream>

using namespace mitsuba;

#if !defined(__OSX__) && !defined(__WINDOWS__)
int main(int argc, char **argv) { std::cout << "qwq" << std::endl; }
#endif