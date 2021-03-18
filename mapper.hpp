#ifndef __MAPPER_H__
#define __MAPPER_H__

#include <string>

namespace wordvec {
    class mapper_t {
    protected:
        union {
            char *rw_data;
            const char *ro_data;
        } m_data;
        off_t m_size = 0;

    public:
        mapper_t(): m_data() {}
        mapper_t(char *_data, off_t _size): m_data(), m_size(_size) {m_data.rw_data = _data;}
        mapper_t(const char *_data, off_t _size): m_data(), m_size(_size) {m_data.ro_data = _data;}
        virtual ~mapper_t() = default;

        inline const char *data() const noexcept {return m_data.ro_data;}
        inline char *data() noexcept {return m_data.rw_data;}
        inline off_t size() const noexcept {return m_size;}
    };

    class string_mapper_t final: public mapper_t {
    public:
        explicit string_mapper_t(const std::string &_source):
                mapper_t(_source.c_str(), static_cast<off_t>(_source.length())) {}

        string_mapper_t(const string_mapper_t &) = delete;
        void operator=(const string_mapper_t &) = delete;
    };

    class file_mapper_t final: public mapper_t {
    private:
        const std::string m_fileName;
        int m_fd = -1;
        const bool m_wr_flag = false;

    public:
        explicit file_mapper_t(const std::string &_fileName, bool _wr_flag = false, off_t _size = 0);
        ~file_mapper_t() final;


        file_mapper_t(const file_mapper_t &) = delete;
        void operator=(const file_mapper_t &) = delete;
    };
}

#endif
