cmake_minimum_required(VERSION 3.5)

project(evsim LANGUAGES CXX CUDA)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/glm)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/glm)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor-blas)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor-blas)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor-io)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtensor-io)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
#add_compile_options(-DXTENSOR_GLIBCXX_USE_CXX11_ABI=1)
add_executable(evsim main.cpp evc.cu shader.cpp model.cpp framebuffer.cpp
    mano.cpp manotypes.cpp manoloader.cpp manogpu.cu manomodel.cpp
    manoanimator.cpp metadatawriter.cpp manotexture.cpp texture.cpp config.cpp)

# Find and link SDL2
find_package(SDL2 REQUIRED)
target_include_directories(evsim PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(evsim ${SDL2_LIBRARIES})

# Find and link SDL2_image
find_package(SDL2_image REQUIRED)
target_include_directories(evsim PRIVATE ${SDL2_IMAGE_INCLUDE_DIRS})
target_link_libraries(evsim ${SDL2_IMAGE_LIBRARIES})

find_package(GLEW REQUIRED)
target_include_directories(evsim PRIVATE ${GLEW_INCLUDE_DIRS})
target_link_libraries(evsim ${GLEW_LIBRARIES})

find_package(OpenGL REQUIRED)
target_include_directories(evsim PRIVATE ${OPENGL_INCLUDE_DIRS})
target_link_libraries(evsim ${OPENGL_LIBRARY})

find_package(GLM REQUIRED)
target_include_directories(evsim PRIVATE ${GLM_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-io REQUIRED)
#target_include_directories(evsim PRIVATE ${xtensor_INCLUDE_DIRS})
#target_link_libraries(evsim ${xtensor_LIBRARY})
target_link_libraries(evsim xtensor)
target_link_libraries(evsim xtensor-blas)
target_link_libraries(evsim z)
target_link_libraries(evsim CUDA::cublas)
target_link_libraries(evsim CUDA::curand)
target_link_libraries(evsim stdc++fs)
