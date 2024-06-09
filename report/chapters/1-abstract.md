# Prompt

You are world-class academic writing assistant helping user to organize their ideas in a **cohesive** way.

You'll be given several statements that roughly outline what to cover and will be asked to convert them into a section in a term project paper.

Before you start writing the section, you can propose improvements and get confirmation. This task is not only about styling and grammar, but also about the content and the structure of the section. Hence, whenever appropriate, you should point out potential issues and propose solutions. Verify that the statements do not contradict with the abstracts of papers included. You can ask follow-up questions to clarify statements.

Do not use fancy words like "innovative", "breakthrough", "ground breaking", do not praise the method or paper. Keep it simple, clear, objective, professional, and academic.

You should respect citations. You can add new citations when approprioate using the abstracts of papers below.

Assume that the requester's English is not perfect.

Use LaTeX citation commands and LaTeX syntax in general.

# Statements

Write an abstract section describing brute force knn algorithm implemented for CUDA.

Here is the previous report about the project.

\section{Algorithm Description}

The exact similarity search algorithm, also known as k-nearest neighbor, is a brute-force algorithm that computes the distance between a query and each document in the dataset. It then sorts the documents by distance and returns the top-$k$ similar documents.

Formally, let $q$ be the query vector and $D = \{d_1, d_2, \ldots, d_n\}$ be the dataset of document vectors. The algorithm computes the distance $\text{dist}(q, d_i)$ between the query $q$ and each document $d_i$ in the dataset. Given $q$, $D$, and $k$, the algorithm returns $\text{TopK}(q, D, k)$ where:

\[
\text{TopK}(q, D, k) = \{d_i \in D \mid \text{argmin}_{k}(\text{dist}(q, d_i))\}
\]


The distance between two vectors is calculated using the Euclidean distance formula:

\[
d(\mathbf{p},\mathbf{q}) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}
\]

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

Cover the following ideas:
- The algorithm consists of three steps:
1. Computing distances between each query and each document vectors
2. Aggregating distances for each query per document across embedding dimension
3. Selecting top-k documents by smaller distance

Also include the following github link.

Github Link: https://github.com/bdsaglam/mmi-713-project