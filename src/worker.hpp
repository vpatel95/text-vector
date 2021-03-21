#ifndef __WORKER_H__
#define __WORKER_H__

#include <memory>
#include <random>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>

#include "word_vector.hpp"
#include "reader.hpp"
#include "vocabulary.hpp"
#include "huffman.hpp"
#include "nsDistribution.hpp"
#include "downSampling.hpp"

namespace wordvec {
    /**
     * @brief trainThread class - train thread and its local data
     *
     *  trainThread class trains a word2vec model from the specified part of train data set file.
     *  Here are two supported training model algorithms - CBOW and Skip-Gram and two approximation algorithms to
     *  speedup training - Hierarchical Softmax (HS) and Negative Sampling (NS).
     *  It is possible to choose any of the following algorithms combination - CBOW/HS or CBOW/NS or Skip-Gram/HS or
     *  Skip-Gram/NS.
    */
    class trainThread_t final {
    public:
        /**
         * @brief sharedData structure holds all common data used by train threads
        */
        struct sharedData_t final {
            std::shared_ptr<train_setting_t> trainSettings; ///< trainSettings structure
            std::shared_ptr<vocabulary_t> vocabulary; ///< words data
            std::shared_ptr<file_mapper_t> fileMapper; ///< train data file access object
            std::shared_ptr<std::vector<float>> bpWeights; ///< back propagation weights
            std::shared_ptr<std::vector<float>> expTable; ///< exp(x) / (exp(x) + 1) values lookup table
            std::shared_ptr<huffmanTree_t> huffmanTree; ///< Huffman tree used by hierarchical softmax
            std::shared_ptr<std::atomic<std::size_t>> processedWords; ///< total words processed by train threads
            std::shared_ptr<std::atomic<float>> alpha; ///< current learning rate
            std::function<void(float, float)> progressCallback = nullptr; ///< callback with alpha and training percent
        };

    private:
        sharedData_t m_sharedData;

        std::random_device m_randomDevice;
        std::mt19937_64 m_randomGenerator;
        std::uniform_int_distribution<short> m_rndWindowShift;
        std::unique_ptr<downSampling_t> m_downSampling;
        std::unique_ptr<nsDistribution_t> m_nsDistribution;
        std::unique_ptr<std::vector<float>> m_hiddenLayerVals;
        std::unique_ptr<std::vector<float>> m_hiddenLayerErrors;
        std::unique_ptr<word_reader_t<file_mapper_t>> m_wordReader;
        std::unique_ptr<std::thread> m_thread;

    public:
        /**
         * Constructs train thread local data
         * @param _id thread ID, starting from 0
         * @param _sharedData sharedData object instantiated outside of the thread
        */
        trainThread_t(uint8_t _id, const sharedData_t &_sharedData);

        /**
         * Launchs the thread
         * @param[out] _trainMatrix - train model matrix
        */
        void launch(std::vector<float> &_trainMatrix) noexcept {
            m_thread.reset(new std::thread(&trainThread_t::worker, this, std::ref(_trainMatrix)));
        }
        /// Joins to the thread
        void join() noexcept {
            return m_thread->join();
        }

    private:
        void worker(std::vector<float> &_trainMatrix) noexcept;

        inline void cbow(const std::vector<const vocabulary_t::wordData_t *> &_sentence,
                         std::vector<float> &_trainMatrix) noexcept;
        inline void skipGram(const std::vector<const vocabulary_t::wordData_t *> &_sentence,
                             std::vector<float> &_trainMatrix) noexcept;
        inline void  hierarchicalSoftmax(std::size_t _index,
                                         std::vector<float> &_hiddenLayer,
                                         std::vector<float> &_trainLayer, std::size_t _trainLayerShift) noexcept;
        inline void negativeSampling(std::size_t _index,
                                     std::vector<float> &_hiddenLayer,
                                     std::vector<float> &_trainLayer, std::size_t _trainLayerShift) noexcept;
    };

}

#endif
