#include "vocabulary.hpp"
#include "reader.hpp"

namespace wordvec {
    vocabulary_t::vocabulary_t(std::shared_ptr<file_mapper_t> &_trainWordsMapper,
                               std::shared_ptr<file_mapper_t> &_stopWordsMapper,
                               const std::string &_delims,
                               const std::string &_eos,
                               uint16_t _minFreq,
                               w2vModel_t::vocabularyProgressCallback_t _progressCallback,
                               w2vModel_t::vocabularyStatsCallback_t _statsCallback) noexcept: m_words() {
        // load stop-words
        std::vector<std::string> stopWords;
        if (_stopWordsMapper) {
            word_reader_t<file_mapper_t> wordReader(*_stopWordsMapper, _delims, _eos);
            std::string word;
            while (wordReader.next_word(word)) {
                stopWords.push_back(word);
            }
        }

        // load words and calculate their frequencies
        struct tmpWordData_t {
            std::size_t frequency = 0;
            std::string word;
        };
        std::unordered_map<std::string, tmpWordData_t> tmpWords;
        off_t progressOffset = 0;
        if (_trainWordsMapper) {
            word_reader_t<file_mapper_t> wordReader(*_trainWordsMapper, _delims, _eos);
            std::string word;
            while (wordReader.next_word(word)) {
                if (word.empty()) {
                    word = "</s>";
                }
                auto &i = tmpWords[word];
                if (i.frequency == 0) {
                    i.word = word;
                }
                i.frequency++;
                m_totalWords++;

                if (_progressCallback != nullptr) {
                    if (wordReader.offset() - progressOffset >= _trainWordsMapper->size() / 10000 - 1) {
                        _progressCallback(static_cast<float>(wordReader.offset())
                                          / _trainWordsMapper->size() * 100.0f);
                        progressOffset = wordReader.offset();
                    }
                }
            }
        }

        // remove stop words from the words set
        for (auto &i:stopWords) {
            tmpWords.erase(i);
        }

        // remove sentence delimiter from the words set
        {
            std::string word = "</s>";
            auto i = tmpWords.find(word);
            if (i != tmpWords.end()) {
                m_totalWords -= i->second.frequency;
                tmpWords.erase(i);
            }
        }

        // prepare vector sorted by word frequencies
        std::vector<std::pair<std::string, std::size_t>> wordsFreq;
        // delimiter is the first word
        wordsFreq.emplace_back(std::pair<std::string, std::size_t>("</s>", 0LU));
        for (auto const &i:tmpWords) {
            if (i.second.frequency >= _minFreq) {
                wordsFreq.emplace_back(std::pair<std::string, std::size_t>(i.first, i.second.frequency));
                m_trainWords += i.second.frequency;
            }
        }

        // sorting, from more frequent to less frequent, skip delimiter </s> (first word)
        if (wordsFreq.size() > 1) {
            std::sort(wordsFreq.begin() + 1, wordsFreq.end(), [](const std::pair<std::string, std::size_t> &_what,
                                                                 const std::pair<std::string, std::size_t>&_with) {
                return _what.second > _with.second;
            });
            // make delimiter frequency more then the most frequent word
            wordsFreq[0].second = wordsFreq[1].second + 1;
            // restore sentence delimiter
            auto &i = tmpWords["</s>"];
            i.word = "</s>";
            i.frequency = wordsFreq[0].second;
        }
        // fill index values
        for (std::size_t i = 0; i < wordsFreq.size(); ++i) {
            auto &w = tmpWords[wordsFreq[i].first];
            m_words[wordsFreq[i].first] = wordData_t(i, w.frequency);
        }

        if (_statsCallback != nullptr) {
            _statsCallback(m_words.size(), m_trainWords, m_totalWords);
        }
    }
}
