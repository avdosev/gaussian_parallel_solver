# CMakeList.txt: проект CMake для gaus; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

find_package(OpenMP REQUIRED)
# Добавьте источник в исполняемый файл этого проекта.
add_executable (gaus "gaus.h" "main.cpp")

target_link_libraries(gaus PRIVATE OpenMP::OpenMP_CXX)
