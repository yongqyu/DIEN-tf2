import numpy as np
import tensorflow as tf

def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def auc_arr(score_p, score_n):
    score_arr = []
    for s in score_p.numpy():
        score_arr.append([0, 1, s])
    for s in score_n.numpy():
        score_arr.append([1, 0, s])
    return score_arr

def eval(model, test_data):
    auc_sum = 0.0
    score_arr = []
    for u, i, j, hist_i, sl in test_data:
        p_out, p_logit = model(u,i,hist_i,sl)
        n_out, n_logit = model(u,j,hist_i,sl)
        mf_auc = tf.reduce_sum(tf.cast(p_out>n_out, dtype=tf.float32))

        score_arr += auc_arr(p_logit, n_logit)
        auc_sum += mf_auc
    test_gauc = auc_sum / len(test_data)
    auc = calc_auc(score_arr)
    return test_gauc, auc

def sequence_mask(lengths, maxlen=None, dtype=tf.bool):
    """Returns a mask tensor representing the first N positions of each cell.

    If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
    dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                    #  [True, True, True, False, False],
                                    #  [True, True, False, False, False]]
    tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                      #   [True, True, True]],
                                      #  [[True, True, False],
                                      #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor, all its values <= maxlen.
        maxlen: scalar integer tensor, size of last dimension of returned tensor.
            Default is the maximum value in `lengths`.
        dtype: output type of the resulting tensor.
        name: name of the op.

    Returns:
        A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
    Raises:
        ValueError: if `maxlen` is not a scalar.
    """
    # lengths = lengths.numpy()

    if maxlen is None:
        maxlen = max(lengths)
    # else:
    #     maxlen = maxlen
    # if maxlen.get_shape().ndims is not None and maxlen.get_shape().ndims != 0:
    #     raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = range(maxlen)
    # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
    # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
    matrix = np.expand_dims(lengths, -1)
    result = row_vector < matrix

    if dtype is None:
        return tf.convert_to_tensor(result)
    else:
        return tf.cast(tf.convert_to_tensor(result), dtype)
