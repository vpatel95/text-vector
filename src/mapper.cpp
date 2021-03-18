#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <stdexcept>
#include <cstring>

#include "mapper.hpp"

namespace wordvec {
    file_mapper_t::file_mapper_t(const std::string &_fileName, bool _wr_flag, off_t _size):
            mapper_t(), m_fileName(_fileName), m_wr_flag(_wr_flag) {

        if (m_wr_flag) {
            m_size = _size;
        }

        // open file
        m_fd = ::open(m_fileName.c_str(), m_wr_flag?(O_RDWR | O_CREAT):O_RDONLY, 0600);
        if (m_fd < 0) {
            std::string err = std::string("fileMapper: ") + _fileName + " - " + std::strerror(errno);
            throw std::runtime_error(err);
        }

        // get file size
        struct stat fst{};
        if (fstat(m_fd, &fst) < 0) {
            std::string err = std::string("fileMapper: ") + _fileName + " - " + std::strerror(errno);
            throw std::runtime_error(err);
        }

        if (!m_wr_flag) {
            if (fst.st_size <= 0) {
                throw std::runtime_error(std::string("fileMapper: file ") + _fileName + " is empty, nothing to read");
            }

            m_size = fst.st_size;
        } else {
            if (ftruncate(m_fd, m_size) == -1) {
                std::string err = std::string("fileMapper: ") + _fileName + " - " + std::strerror(errno);
                throw std::runtime_error(err);
            }
        }

        // map file to memory
        m_data.rw_data = static_cast<char *>(mmap(nullptr, static_cast<size_t>(m_size),
                                                 m_wr_flag?(PROT_READ | PROT_WRITE):PROT_READ , MAP_SHARED,
                                                 m_fd, 0));
        if (m_data.rw_data == static_cast<char *>(MAP_FAILED)) {
            std::string err = std::string("fileMapper: ") + _fileName + " - " + std::strerror(errno);
            throw std::runtime_error(err);
        }
    }

    file_mapper_t::~file_mapper_t() {
#if defined(sun) || defined(__sun)
        munmap(m_data.rw_data, static_cast<size_t>(m_size));
#else
        munmap(reinterpret_cast<void *>(m_data.rw_data), static_cast<size_t>(m_size));
#endif
        close(m_fd);
    }
}
