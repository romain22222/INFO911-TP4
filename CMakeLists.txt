cmake_minimum_required(VERSION 3.0)

project(tp4)

# Define your source files
file(GLOB SOURCES "src/*.cpp")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Create an executable target
add_executable(tp4 ${SOURCES})

# Link the OpenCV libraries to your executable
target_link_libraries(tp4 ${OpenCV_LIBS})