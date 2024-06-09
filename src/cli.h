#include <string>
#include <cstdlib>

struct Params {
    int N = 1000; // number of documents
    int Q = 100; // number of queries
};

void parseCommandLine(int argc, char* argv[], Params& params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) {
            params.N = std::atol(argv[++i]);
        } else if (arg == "--Q" && i + 1 < argc) {
            params.Q = std::atol(argv[++i]);
        }
    }
}
