project (wv-tools)
cmake_minimum_required(VERSION 3.1)

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${PROJECT_INCLUDE_DIR}")

link_directories(${LIBRARY_OUTPUT_PATH})

set(TRAINER_NAME wv-trainer)
set(TRAINER_SRCS ${PROJECT_SOURCE_DIR}/trainer.cpp)
add_executable(${TRAINER_NAME} ${TRAINER_SRCS})
target_link_libraries(${TRAINER_NAME} word-vec ${LIBS})

set(DISTANCE_NAME wv-dist)
set(DISTANCE_SRCS ${PROJECT_SOURCE_DIR}/distance.cpp)
add_executable(${DISTANCE_NAME} ${DISTANCE_SRCS})
target_link_libraries(${DISTANCE_NAME} word-vec ${LIBS})

set(ANALOGY_NAME wv-analogy)
set(ANALOGY_SRCS ${PROJECT_SOURCE_DIR}/analogy.cpp)
add_executable(${ANALOGY_NAME} ${ANALOGY_SRCS})
target_link_libraries(${ANALOGY_NAME} word-vec ${LIBS})

set(ACCURACY_NAME wv-accuracy)
set(ACCURACY_SRCS ${PROJECT_SOURCE_DIR}/accuracy.cpp)
add_executable(${ACCURACY_NAME} ${ACCURACY_SRCS})
target_link_libraries(${ACCURACY_NAME} word-vec ${LIBS})

install(TARGETS ${TRAINER_NAME} DESTINATION bin)
install(TARGETS ${DISTANCE_NAME} DESTINATION bin)
install(TARGETS ${ANALOGY_NAME} DESTINATION bin)
install(TARGETS ${ACCURACY_NAME} DESTINATION bin)
