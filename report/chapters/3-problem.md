\begin{document}

\maketitle

\section{Problem}

The exact similarity search algorithm, also known as k-nearest neighbor, is a brute-force algorithm that computes the distance between a query and each document in the dataset. It then sorts the documents by distance and returns the top-$k$ similar documents.

Formally, let $q$ be the query vector and $D = \{d_1, d_2, \ldots, d_n\}$ be the dataset of document vectors. The algorithm computes the distance $\text{dist}(q, d_i)$ between the query $q$ and each document $d_i$ in the dataset. Given $q$, $D$, and $k$, the algorithm returns $\text{TopK}(q, D, k)$ where:

\[
\text{TopK}(q, D, k) = \{d_i \in D \mid \text{argmin}_{k}(\text{dist}(q, d_i))\}
\]


The distance between two vectors is calculated using the Euclidean distance formula:

\[
d(\mathbf{p},\mathbf{q}) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}
\]
