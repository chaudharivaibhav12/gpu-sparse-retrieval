# Retrieval concepts

SPLADE represents a query and passage as sparse term-weight vectors. Their score is a weighted dot product over shared terms. An inverted index groups matching passages by term; a forward index groups terms by passage.

Exhaustive scoring computes every relevant dot product. Block-max pruning groups passages into disjoint blocks and stores each term's maximum impact per block. Summing query-weighted maxima produces a conservative score bound. Once k distinct scored passages establish a threshold, a block whose bound is strictly smaller can be skipped. Equality must be handled consistently with the result tie rule.

Top-k selects the k greatest scores, breaking ties by original numeric passage ID. Disjoint partitions can return their local top-k and merge them into the global top-k under that same total order. Selection scratch and transfer costs belong in the measured system cost.

The current prototype uses exact byte-valued document impacts and uint64 accumulation. Prepared query weights follow the pinned BMP float32/ceil cap32 transform. This defines an integerized scoring contract, not equivalence to original model FP32 retrieval.

The offline pilot checks candidate coverage and logical memory touches. The initial CUDA scorer performs document scoring on GPU and ranking on CPU. GPU selection, block pruning and progressive bounds remain implementation work.
