Johnson, Douze, and Jégou (2019) [1] introduce an innovative approach for accelerating vector search on GPUs, significantly improving upon the state of the art by proposing optimized designs for brute-force, approximate, and compressed-domain search. Their method achieves up to 8.5 times faster performance than previous GPU-based solutions, demonstrating its effectiveness across various similarity search scenarios, including the construction of high-accuracy k-NN graphs on large-scale datasets.

Zhao, Tan, and Li (2020) [2] introduced SONG, a framework that addresses the challenge of executing graph-based Approximate Nearest Neighbor (ANN) search on GPUs by segmenting the search algorithm into three distinct stages for enhanced parallelism. This approach not only capitalizes on the computational strengths of GPUs but also incorporates ANN-specific optimizations to minimize dynamic memory allocations, thus achieving a remarkable speedup of 50-180x over HNSW—a leading CPU-based ANN method—while also surpassing the performance of Faiss, a well-known GPU-accelerated ANN platform, across multiple datasets.

Groh et al. (2023) [3] present GGNN, a GPU-friendly search structure that addresses the bottleneck in the construction of index structures for ANN search. By leveraging nearest neighbor graphs and information propagation strategies optimized for GPU architectures, GGNN not only accelerates the hierarchical index building process but also enhances query performance. It demonstrates significant advancements over existing CPU and GPU systems in build-time efficiency, search speed, and accuracy, setting a new benchmark in the field of ANN search.


@article{Johnson2019,
  author    = {Johnson, J. and Douze, M. and Jégou, H.},
  title     = {Billion-scale similarity search with GPUs},
  journal   = {IEEE Transactions on Big Data},
  volume    = {7},
  number    = {3},
  pages     = {535--547},
  year      = {2019},
  publisher = {IEEE}
}

@inproceedings{Zhao2020,
  author    = {Zhao, W. and Tan, S. and Li, P.},
  title     = {SONG: Approximate Nearest Neighbor Search on GPU},
  booktitle = {2020 IEEE 36th International Conference on Data Engineering (ICDE)},
  pages     = {1033--1044},
  year      = {2020},
  organization = {IEEE}
}

@article{Groh2023,
  author    = {Groh, F. and Ruppert, L. and Wieschollek, P. and Lensch, H. P. A.},
  title     = {GGNN: Graph-Based GPU Nearest Neighbor Search},
  journal   = {IEEE Transactions on Big Data},
  volume    = {9},
  number    = {1},
  pages     = {267--279},
  year      = {2023},
  publisher = {IEEE}
}
