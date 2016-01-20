def inverse_frequency_weights(series):
    vals, counts = np.unique(series, return_counts=True)
    freqs = counts.astype('float')/counts.sum()
    weights = freqs.take(series)
    return weights
