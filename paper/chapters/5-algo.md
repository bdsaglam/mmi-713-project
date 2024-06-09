\section{Algorithm Design}

The algorithm can be split into following steps: 
\begin{enumerate}
\item Calculate distances between query and documents
\item Selection documents within smallest k distances
\end{enumerate}

\subsection{Distance computation}
Let $N$ be the vector size of query and document, the set of documents can be represented as 2D array of $D \times N$. The distances between query and set of documents is calculated in parallel by distributing document vectors on a 2D grid of $D \times N$. In each thread, The kernel computes the square of difference between $q_x$ element and $d_y,_x$ element. Then, the kernel sums the differences for all elements for a document. For summation, we explore an implementation with loop and an implementation with parallel reduction.

There are two options to read the elements of document vectors within threads: GPU global memory and textures. As the document vectors are to be read-only, it is suitable for texture which might speedup memory reads \cite{garcia-2008-phd-thesis}. Both alternatives are to be explored in the project.

\subsection{K-selection}

After computing distances, the algorithm selects the smallest k distances together with their indices, which is used for identifying the documents. There are many k-selection algorithms for CUDA in the literature but in this project, block-select algorithm \cite{douze2024faiss} is explored. Block-select algorithm divides an array into multiple chunks which are distributed on blocks. Each block finds the top-k elements in the chunk using shared memory and heap data structure. Then, all top-k elements from different blocks are merged and the final top-k elements are found. 

\section{Benchmarking}

MuSiQue-Ans \cite{trivedi-etal-2022-musique} is a multi-hop question answering dataset consisting of more than 25000 records. Each record includes around 20 paragraphs, a multi-hop question, the answer for the question, and the indices for paragraphs supporting the answer. In this project, the algorithm is to be benchmarked by using the paragraphs as documents and questions as search queries. An open-source embedding model such as "sentence-transformers/all-MiniLM-L6-v2" is used to convert document and query texts into vectors.

The following parameters are to explored in the benchmarking:
\begin{itemize}
    \item Number of documents
    \item Block size
    \item Memory type \{global, texture\}
    \item Summation algorithm type \{for loop, parallel reduction\}
    \item K-selection algorithm \{block-select, sort on CPU\}
\end{itemize}

The throughput of algorithm, which is defined as number of queries / time, is to be measured for each experiment. The algorithm is to be compared to CPU implementation.

\bibliographystyle{plain}
\bibliography{main.bib}
\end{document}

@article{trivedi-etal-2022-musique,
title={♫ MuSiQue: Multihop Questions via Single-hop Question Composition},
author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
journal={Transactions of the Association for Computational Linguistics},
volume={10},
pages={539--554},
year={2022},
publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…}
}

@phdthesis{garcia-2008-phd-thesis,
  title={Suivi d'objets d'intérêt dans une séquence d'images : des points saillants aux mesures statistiques},
  author={Garcia, Vincent},
  year={2008},
  school={Universit{\'e} de Nice - Sophia Antipolis},
  address={Sophia Antipolis, France},
  month={December}
}

@inproceedings{batcher1968sorting,
  title={Sorting networks and their applications},
  author={Batcher, Kenneth E},
  booktitle={AFIPS '68 (Fall, part I): Proceedings of the December 9-11, 1968, fall joint computer conference, part I},
  pages={307--314},
  year={1968},
  organization={ACM}
}

@article{johnson2019billion,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}

@article{douze2024faiss,
  title={The faiss library},
  author={Douze, Matthijs and Guzhva, Alexandr and Deng, Chengqi and Johnson, Jeff and Szilvasy, Gergely and Mazar{\'e}, Pierre-Emmanuel and Lomeli, Maria and Hosseini, Lucas and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2401.08281},
  year={2024}
}