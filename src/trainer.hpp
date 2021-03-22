#ifndef __TRAINER_H__
#define __TRAINER_H__

#include <memory>
#include <vector>
#include <functional>

#include "word_vector.hpp"
#include "reader.hpp"
#include "vocabulary.hpp"
#include "worker.hpp"

namespace wordvec {
    /**
     * @brief trainer class of word2vec model
     *
     * trainer class is responsible for train-specific data instantiation, train threads control and
     * train process itself.
    */
    class trainer_t {
    private:
        std::size_t m_matrixSize = 0;
        std::vector<std::unique_ptr<trainThread_t>> m_threads;

    public:
        /**
         * Constructs a trainer object
         * @param _trainSettings trainSattings object
         * @param _vocabulary vocabulary object
         * @param _fileMapper fileMapper object related to a train data set file
         * @param _progressCallback callback function to be called on each new 0.01% processed train data
        */
        trainer_t(const std::shared_ptr<train_setting_t> &_trainSettings,
                  const std::shared_ptr<vocabulary_t> &_vocabulary,
                  const std::shared_ptr<file_mapper_t> &_fileMapper,
                  std::function<void(float, float)> _progressCallback);

        /**
         * Runs training process
         * @param[out] _trainMatrix train model matrix
        */
        void operator()(std::vector<float> &_trainMatrix) noexcept;
    };
}

#endif
