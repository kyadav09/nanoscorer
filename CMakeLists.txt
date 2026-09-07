cmake_minimum_required(VERSION 3.16)
project(simd_inference_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(engine_headers INTERFACE)
target_include_directories(engine_headers INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Release: the numbers you'd actually report.
add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE engine_headers)
target_compile_options(test_order_book PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

# Sanitized: the build you run in CI / before trusting the release build.
add_executable(test_order_book_asan tests/test_order_book.cpp)
target_link_libraries(test_order_book_asan PRIVATE engine_headers)
target_compile_options(test_order_book_asan PRIVATE -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(test_order_book_asan PRIVATE -fsanitize=address,undefined)

add_executable(test_quickscorer tests/test_quickscorer.cpp)
target_link_libraries(test_quickscorer PRIVATE engine_headers)
target_compile_options(test_quickscorer PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

add_executable(test_quickscorer_asan tests/test_quickscorer.cpp)
target_link_libraries(test_quickscorer_asan PRIVATE engine_headers)
target_compile_options(test_quickscorer_asan PRIVATE -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(test_quickscorer_asan PRIVATE -fsanitize=address,undefined)

add_executable(test_quickscorer_avx2 tests/test_quickscorer_avx2.cpp)
target_link_libraries(test_quickscorer_avx2 PRIVATE engine_headers)
target_compile_options(test_quickscorer_avx2 PRIVATE -O3 -mavx2 -DNDEBUG -Wall -Wextra)

add_executable(test_quickscorer_avx2_asan tests/test_quickscorer_avx2.cpp)
target_link_libraries(test_quickscorer_avx2_asan PRIVATE engine_headers)
target_compile_options(test_quickscorer_avx2_asan PRIVATE -O1 -g -mavx2 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(test_quickscorer_avx2_asan PRIVATE -fsanitize=address,undefined)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE engine_headers)
target_compile_options(test_matmul PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)

add_executable(test_matmul_asan tests/test_matmul.cpp)
target_link_libraries(test_matmul_asan PRIVATE engine_headers)
target_compile_options(test_matmul_asan PRIVATE -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(test_matmul_asan PRIVATE -fsanitize=address,undefined)

find_package(Threads REQUIRED)

# add_executable(test_pipeline tests/test_pipeline.cpp)
# target_link_libraries(test_pipeline PRIVATE engine_headers Threads::Threads)
# target_compile_options(test_pipeline PRIVATE -O2 -march=native -DNDEBUG -Wall -Wextra)

# add_executable(test_pipeline_tsan tests/test_pipeline.cpp)
# target_link_libraries(test_pipeline_tsan PRIVATE engine_headers Threads::Threads)
# target_compile_options(test_pipeline_tsan PRIVATE -O1 -g -fsanitize=thread -Wall -Wextra)
# target_link_options(test_pipeline_tsan PRIVATE -fsanitize=thread)

enable_testing()
add_test(NAME order_book_asan COMMAND test_order_book_asan)
add_test(NAME order_book_release COMMAND test_order_book)
add_test(NAME quickscorer_asan COMMAND test_quickscorer_asan)
add_test(NAME quickscorer_release COMMAND test_quickscorer)
add_test(NAME quickscorer_avx2_asan COMMAND test_quickscorer_avx2_asan)
add_test(NAME quickscorer_avx2_release COMMAND test_quickscorer_avx2)
add_test(NAME matmul_asan COMMAND test_matmul_asan --correctness-only)
add_test(NAME matmul_release COMMAND test_matmul)
# add_test(NAME pipeline_tsan COMMAND test_pipeline_tsan)
# add_test(NAME pipeline_release COMMAND test_pipeline)