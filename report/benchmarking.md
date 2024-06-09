
The CPU and GPU implementations of kNN algorithm are compared by execution time for different number of documents and number of queries. The embedding dimension is set as 512, and the number of documents to match (top-k) is set as 10. Each experiment is repeated three times to mitigate variance.


## Observations
### Execution time vs number of docs
number_of_queries=10
GPU execution time stays flat while CPU execution time linearly increasing with number of docs. CPU implementation is faster except when number of docs >= 10000. 

number_of_queries=100
GPU execution time stays flat while CPU execution time linearly increasing with number of docs. GPU implementation is faster except when number of docs <= 100. 

### Execution time vs number of queries
number_of_docs=100
GPU execution time stays flat while CPU execution time linearly increasing with number of docs. CPU is faster.

number_of_docs=1000
GPU execution time stays flat while CPU execution time linearly increasing with number of docs. CPU is faster except high number of queries (>1000).

## Conclusion
It can be concluded that on small data, the execution time for GPU implementations is dominated by memory copy overhead between host and device; hence, the speedup of computation on GPU doesn't have much impact on execution time. However, when the data becomes larger (n_docs>=10000 or n_queries>=1000), the time spent on computation becomes predominant; hence, making GPU implementation faster due to parallelization of computation.
