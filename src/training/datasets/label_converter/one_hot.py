import numpy as np


def one_hot(labels, num_classes, dim=1):
    """
    Convert a numpy array into one-hot format.

    Args:
        labels: input numpy array of integers to be converted into the 'one-hot' format.
        num_classes: number of output channels.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Returns:
        numpy array of one-hot encoded format.
    """

    # Ensure labels are integer type
    labels = labels.astype(np.int64)

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = np.reshape(labels, shape)

    if labels.shape[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    # Prepare the shape for the output array
    sh = list(labels.shape)
    sh[dim] = num_classes

    # Initialize output array with zeros
    o = np.zeros(sh, dtype=np.float32)

    # Use numpy.put_along_axis to achieve the one-hot encoding
    np.put_along_axis(o, labels, 1, axis=dim)

    return o


if __name__ == "__main__":
    mask = np.random.randint(0, 8, size=(1, 256, 256))
    one_hot_label = one_hot(mask, 9, dim=0)
    print()