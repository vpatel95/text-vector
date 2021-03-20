#ifndef __WORD_VECTOR_H__
#define __WORD_VECTOR_H__

#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace wordvec {
    struct train_setting_t final {
        uint16_t min_freq = 5;
        uint16_t size = 100;
        uint8_t window = 5;
        uint16_t table_sz = 1000;
        uint8_t table_max = 6;
        float sample = 1e-3f;
        bool with_hs = false;
        uint8_t negative = 5;
        uint8_t threads = 4;
        uint8_t iterations = 5;
        float alpha = 0.05f;
        bool with_sg = false;
        std::string delims = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r";
        std::string eos = ".\n?!";
        train_setting_t() = default;
    };

    class vector_t: public std::vector<float> {
        public:
            vector_t(): std::vector<float>() {}

            explicit vector_t(std::size_t _size): std::vector<float>(_size, 0.0f) {}


            explicit vector_t(const std::vector<float> &_vector): std::vector<float>(_vector) {}


            vector_t &operator+=(const vector_t &_from) {
                if (this != &_from) {
                    assert(size() == _from.size());

                    for (std::size_t i = 0; i <  size(); ++i) {
                        (*this)[i] += _from[i];
                    }
                    float med = 0.0f;
                    for (auto const &i:(*this)) {
                        med += i * i;
                    }
                    if (med <= 0.0f) {
                        throw std::runtime_error("word2vec: can not create vector");
                    }
                    med = std::sqrt(med / this->size());
                    for (auto &i:(*this)) {
                        i /= med;
                    }
                }

                return *this;
            }


            vector_t &operator-=(const vector_t &_from) {
                if (this != &_from) {
                    assert(size() == _from.size());

                    for (std::size_t i = 0; i < size(); ++i) {
                        (*this)[i] -= _from[i];
                    }
                    float med = 0.0f;
                    for (auto const &i:(*this)) {
                        med += i * i;
                    }
                    if (med <= 0.0f) {
                        throw std::runtime_error("word2vec: can not create vector");
                    }
                    med = std::sqrt(med / this->size());
                    for (auto &i:(*this)) {
                        i /= med;
                    }
                }

                return *this;
            }

            friend vector_t operator+(vector_t _what, const vector_t &_with) {
                _what += _with;
                return _what;
            }

            friend vector_t operator-(vector_t _what, const vector_t &_with) {
                _what -= _with;
                return _what;
            }
    };

    template <class key_t>
        class model_t {
            protected:
                using map_t = std::unordered_map<key_t, vector_t>;

                map_t m_map;
                uint16_t m_vec_sz = 0;
                std::size_t m_map_sz = 0;
                mutable std::string m_err_msg;

                const std::string wrong_format_err = "model: wrong model file format";

            private:
                struct nearest_cmp_t final {
                    inline bool operator()(const std::pair<key_t, float> &_left,
                            const std::pair<key_t, float> &_right) const noexcept {
                        return _left.second > _right.second;
                    }
                };

            public:
                model_t(): m_map(), m_err_msg() {}
                virtual ~model_t() = default;

                const map_t &map() {return m_map;}

                virtual bool save(const std::string &_model_file) const noexcept = 0;
                virtual bool load(const std::string &_model_file) noexcept = 0;

                inline const vector_t *vector(const key_t &_key) const noexcept {
                    auto const &i = m_map.find(_key);
                    if (i != m_map.end()) {
                        return &i->second;
                    }

                    return nullptr;
                }

                inline float distance(const vector_t &_what, const vector_t &_with) const noexcept {
                    assert(m_vec_sz == _what.size());
                    assert(m_vec_sz == _with.size());

                    float ret = 0.0f;
                    for (uint16_t i = 0; i < m_vec_sz; ++i) {
                        ret += _what[i] * _with[i];
                    }
                    if (ret > 0.0f) {
                        return  std::sqrt(ret / m_vec_sz);
                    }
                    return 0.0f;
                }

                inline void nearest(const vector_t &_vec,
                        std::vector<std::pair<key_t, float>> &_nearest,
                        std::size_t _amount,
                        float _minDistance = 0.0f) const noexcept {
                    assert(m_vec_sz == _vec.size());

                    _nearest.clear();

                    std::priority_queue<std::pair<key_t, float>,
                        std::vector<std::pair<key_t, float>>,
                        nearest_cmp_t> nearestVecs;

                    float entryLevel = 0.0f;
                    for (auto const &i:m_map) {
                        auto match = distance(_vec, i.second);
                        if ((match > 0.9999f) || (match < _minDistance)) {
                            continue;
                        }
                        if (match > entryLevel) {
                            nearestVecs.emplace(std::pair<key_t, float>(i.first, match));
                            if (nearestVecs.size() > _amount) {
                                nearestVecs.pop();
                                entryLevel = nearestVecs.top().second;
                            }
                        }
                    }

                    auto nSize = nearestVecs.size();
                    _nearest.resize(nSize);
                    for (auto j = nSize; j > 0; --j) {
                        _nearest[j - 1] = nearestVecs.top();
                        nearestVecs.pop();
                    }
                }


                inline uint16_t vectorSize() const noexcept {return m_vec_sz;}

                inline std::size_t modelSize() const noexcept {return m_map_sz;}

                inline std::string errMsg() const noexcept {return m_err_msg;}
        };

    class w2vModel_t: public model_t<std::string> {
        public:

            using vocabularyProgressCallback_t = std::function<void(float)>;

            using vocabularyStatsCallback_t = std::function<void(std::size_t, std::size_t, std::size_t)>;

            using trainProgressCallback_t = std::function<void(float, float)>;

        public:

            w2vModel_t(): model_t<std::string>() {}

            bool train(const train_setting_t &_trainSettings,
                    const std::string &_trainFile,
                    const std::string &_stopWordsFile,
                    vocabularyProgressCallback_t _vocabularyProgressCallback,
                    vocabularyStatsCallback_t _vocabularyStatsCallback,
                    trainProgressCallback_t _trainProgressCallback) noexcept;


            bool save(const std::string &_model_file) const noexcept override;

            bool load(const std::string &_model_file) noexcept override;
    };

    class d2vModel_t: public model_t<std::size_t> {
        public:
            explicit d2vModel_t(uint16_t _vectorSize): model_t<std::size_t>() {
                m_vec_sz = _vectorSize;
            }


            void set(std::size_t _id, const vector_t &_vector, bool _checkUnique = false) {
                if (_checkUnique) {
                    for (auto const &i:m_map) {
                        auto match = distance(_vector, i.second);
                        if (match > 0.9999f) {
                            return;
                        }
                    }
                }

                m_map[_id] = _vector;
                m_map_sz = m_map.size();
            }

            void erase(std::size_t _id) {
                m_map.erase(_id);
                m_map_sz = m_map.size();
            }

            bool save(const std::string &_model_file) const noexcept override;

            bool load(const std::string &_model_file) noexcept override;
    };

    class word2vec_t: public vector_t {
        public:
            explicit word2vec_t(const std::unique_ptr<w2vModel_t> &_model): vector_t(_model->vectorSize()) {}

            word2vec_t(const std::unique_ptr<w2vModel_t> &_model, const std::string &_word):
                vector_t(_model->vectorSize()) {
                    auto i = _model->vector(_word);
                    if (i != nullptr) {
                        std::copy(i->begin(), i->end(), begin());
                    }
                }
    };

    class doc2vec_t: public vector_t {
        public:
            doc2vec_t(const std::unique_ptr<w2vModel_t> &_model,
                    const std::string &_doc,
                    const std::string &_delims = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r");
    };

}
#endif
