#include <getopt.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "word_vector.hpp"

int main(int argc, char * const *argv) {
    if (argc != 2) {
        std::cerr << "Usage:" << std::endl
                  << argv[0] << " [word2vec_model_file_name]" << std::endl;
        return 1;
    }

    // create w2v model object
    std::unique_ptr<wordvec::w2vModel_t> model;
    try {
        model.reset(new wordvec::w2vModel_t());
        // load w2v model file
        if (!model->load(argv[1])) {
            throw std::runtime_error(model->errMsg());
        }
    } catch (const std::exception &_e) {
        std::cerr << _e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 2;
    }

    while (true) {
        try {
            std::cout << "Enter word or sentence (EXIT to break): ";
            char buf[4096];
            std::cin.getline(buf, 4095);
            std::string query = buf;
            if (query == "EXIT") {
                break;
            }
            std::cout << std::right << std::setw(19) << "Word" << std::left << std::setw(9) << " Distance" << std::endl;
            std::cout << std::right << std::setw(28) << std::setfill('-') << "-" << std::setfill(' ') << std::endl;

            wordvec::doc2vec_t vec(model, query);
            std::vector<std::pair<std::string, float>> nearests;
            model->nearest(vec, nearests, 30);
            for (auto const &i:nearests) {
                std::cout << std::right << std::setw(19) << i.first << " "
                          << std::left << std::setw(9) << i.second
                          << std::endl;
            }
            std::cout << std::endl;
        } catch (const std::exception &_e) {
            std::cerr << _e.what() << std::endl;
        } catch (...) {
            std::cerr << "unknown error" << std::endl;
        }
    }

    return 0;
}
