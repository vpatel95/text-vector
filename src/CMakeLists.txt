project (word-vec)
cmake_minimum_required(VERSION 3.1)

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${PROJECT_INCLUDE_DIR}")

set(PRJ_SRCS
        ${PROJECT_INCLUDE_DIR}/word_vector.hpp
        ${PROJECT_SOURCE_DIR}/word_vector.cpp
#        ${PROJECT_INCLUDE_DIR}/word2vec.h
#        ${PROJECT_SOURCE_DIR}/c_binding.cpp
        ${PROJECT_SOURCE_DIR}/mapper.cpp
        ${PROJECT_SOURCE_DIR}/vocabulary.hpp
        ${PROJECT_SOURCE_DIR}/vocabulary.cpp
        ${PROJECT_SOURCE_DIR}/huffman.hpp
        ${PROJECT_SOURCE_DIR}/huffman.cpp
        ${PROJECT_SOURCE_DIR}/nsDistribution.hpp
        ${PROJECT_SOURCE_DIR}/nsDistribution.cpp
        ${PROJECT_SOURCE_DIR}/downSampling.hpp
        ${PROJECT_SOURCE_DIR}/trainer.hpp
        ${PROJECT_SOURCE_DIR}/trainer.cpp
        ${PROJECT_SOURCE_DIR}/worker.hpp
        ${PROJECT_SOURCE_DIR}/worker.cpp
        ${ADD_SRCS}
        )

add_library(${PROJECT_NAME} STATIC ${PRJ_SRCS})
target_link_libraries(${PROJECT_NAME} ${LIBS})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${PROJECT_INCLUDE_DIR}/word_vector.hpp DESTINATION include)
install(FILES ${PROJECT_INCLUDE_DIR}/mapper.hpp DESTINATION include)
install(FILES ${PROJECT_INCLUDE_DIR}/reader.hpp DESTINATION include)
