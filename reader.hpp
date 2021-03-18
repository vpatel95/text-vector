#ifndef __READER_H__
#define __READER_H__

#include <string>
#include <cstring>
#include <stdexcept>

#include "mapper.hpp"

namespace wordvec {
    template <class data_mapper_t>
    class word_reader_t final {
    private:
        const data_mapper_t &m_mapper;
        std::string m_delims;
        std::string m_eos;
        const uint16_t m_max_word_len;
        off_t m_offset;
        const off_t m_start;
        const off_t m_stop;
        std::string m_word;
        std::size_t m_pos = 0;
        bool m_last_eos = false;

    public:
        word_reader_t(const data_mapper_t &_mapper,
                     std::string _delims,
                     std::string _eos,
                     off_t _offset = 0, off_t _stop = 0, uint16_t _maxWordLen = 100):
                m_mapper(_mapper),
                m_delims(std::move(_delims)),
                m_eos(std::move(_eos)),
                m_max_word_len(_maxWordLen), m_offset(_offset),
                m_start(m_offset), m_stop((_stop == 0)?_mapper.size() - 1:_stop),
                m_word(m_max_word_len, 0) {

            if (m_stop >= m_mapper.size()) {
                throw std::range_error("wordReader: bounds are out of the file size");
            }
            if (m_offset > m_stop) {
                throw std::range_error("wordReader: offset is out of the bounds");
            }
        }


        word_reader_t(const word_reader_t &) = delete;
        void operator=(const word_reader_t &) = delete;


        inline off_t offset() const noexcept {return m_offset;}


        inline void reset() noexcept {
            m_offset = m_start;
            m_pos = 0;
            m_last_eos = false;
        }

        inline bool next_word(std::string &_word) noexcept {
            while (m_offset <= m_stop) {
                char ch = m_mapper.data()[m_offset++];
                if (m_delims.find(ch) != std::string::npos) {
                    if (m_eos.find(ch) != std::string::npos) {
                        if (m_pos > 0) {
                            m_offset--;
                            m_last_eos = false;
                            break;
                        } else {
                            if (!m_last_eos) {
                                _word.clear();
                                m_last_eos = true;
                                return true;
                            } else {
                                continue;
                            }
                        }
                    }
                    if (m_pos > 0) {
                        m_last_eos = false;
                        break;
                    } else {
                        continue;
                    }
                }
                if (m_pos < m_max_word_len) {
                    m_word[m_pos++] = ch;
                }
            }
            if (m_pos > 0) {
                try {
                    _word.resize(m_pos);
                    std::copy(m_word.data(), m_word.data() + m_pos, &_word[0]);
                } catch (...) {
                    return false;
                }
                m_pos = 0;
                return true;
            }

            return false;
        }
    };
}

#endif
