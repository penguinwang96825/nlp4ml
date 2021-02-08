import numpy as np
import itertools


def by_indexes(iterable):
    output = {}
    for index, key in enumerate(iterable):
        output.setdefault(key, []).append(index)
    return output


def co_occurrence_matrix(corpus, vocabulary, window_size=5):
    """
    Parameters
    ----------
    corpus: list
    vocabulary: set
    window_size: int

    Returns
    -------
    matrix: np.array
    vocabulary_indexes: list

    Examples
    --------
    >>> documents = [["I", "love", "nlp"], ["nlp", "stands", "for", "natural", "language", "processing"]]
    >>> corpus = ["I love nlp", "nlp stands for natural language processing"]
    >>> vocab = set([term for doc in documents for term in doc])
    >>> co_matrix, vocab_idx = co_occurrence_matrix(corpus, vocab, window_size=1)
    >>> pd.DataFrame(co_matrix, index=vocab_idx, columns=vocab_idx)

    References
    ----------
    [1] https://codereview.stackexchange.com/questions/235633/generating-a-co-occurrence-matrix
    """
    def split_tokens(tokens):
        for token in tokens:
            indexs = vocabulary_indexes.get(token)
            if indexs is not None:
                yield token, indexs[0]

    matrix = np.zeros((len(vocabulary), len(vocabulary)), np.float64)
    vocabulary_indexes = by_indexes(vocabulary)

    for sent in corpus:
        tokens = by_indexes(split_tokens(sent.split())).items()
        for ((word_1, x), indexes_1), ((word_2, y), indexes_2) in itertools.permutations(tokens, 2):
            for k in indexes_1:
                for l in indexes_2:
                    if abs(l - k) <= window_size:
                        matrix[x, y] += 1

    return matrix, vocabulary_indexes


def get_ppmi(matrix):
    """
    Tranform a counts matrix to PPMI.
    
    Parameters
    ----------
    matrix: np.array
    
    Returns
    -------
    ret: scipy.sparse.csc_matrix
    """
    total_count = matrix.sum()
    count_words = np.array(matrix.sum(axis=1), dtype=np.float64).flatten()
    ii, jj = matrix.nonzero()
    ij = np.array(matrix[ii, jj], dtype=np.float64).flatten()
    pmi = np.log(ij * total_count / (count_words[ii] * count_words[jj]))
    ppmi = np.maximum(0, pmi)
    ret = scipy.sparse.csc_matrix((ppmi, (ii, jj)), 
                                  shape=matrix.shape,
                                  dtype=np.float64)
    ret.eliminate_zeros()
    return ret

def pretty_print_matrix(M, rows=None, cols=None, dtype=float, float_fmt="{0:.04f}"):
    df = pd.DataFrame(M, index=rows, columns=cols, dtype=dtype)
    old_fmt_fn = pd.get_option('float_format')
    pd.set_option('float_format', lambda f: float_fmt.format(f))
    display(df)
    pd.set_option('float_format', old_fmt_fn)