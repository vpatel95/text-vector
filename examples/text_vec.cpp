#include <iostream>
#include <stdexcept>

#include "word_vector.hpp"

int main(int argc, char * const *argv) {
    if (argc != 2) {
        std::cerr << "Usage:" << std::endl
                  << argv[0] << " [word2vec_model_file_name]" << std::endl;
        return 1;
    }

    std::unique_ptr<wordvec::w2vModel_t> w2vModel;
    try {
        w2vModel.reset(new wordvec::w2vModel_t());
        if (!w2vModel->load(argv[1])) {
            throw std::runtime_error(w2vModel->errMsg());
        }
    } catch (const std::exception &_e) {
        std::cerr << _e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 2;
    }

    try {
        wordvec::word2vec_t king(w2vModel, "king");
        wordvec::word2vec_t man(w2vModel, "man");
        wordvec::word2vec_t woman(w2vModel, "woman");

        wordvec::vector_t result = king - man + woman;


        std::vector<std::pair<std::string, float>> nearest;
        w2vModel->nearest(result, nearest, 10);


        for (auto const &i:nearest) {

            if ((i.first == "king") || (i.first == "man") || (i.first == "woman")) {
                continue;
            }

            std::cout << i.first << ": " << i.second << std::endl;
        }
/*
 * The nearest word to the result vector is "queen", our model works well.
 * Output should looks like -
 * queen: 0.542737
 * princess: 0.499493
 * mother: 0.479788
 * daughter: 0.475785
 * her: 0.470077
 * monarch: 0.464647
 * infanta: 0.461744
 * husband: 0.460482
*/
    } catch (const std::exception &_e) {
        std::cerr << _e.what() << std::endl;
        return 3;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 3;
    }
    return 0;
}
