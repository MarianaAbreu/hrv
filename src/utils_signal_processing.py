import numpy as np
from scipy.stats import pearsonr


def clean_outliers(templates, way='q'):

    # shape of templates
    if len(templates.shape) > 1:
        var = templates.var(axis=1)
    else:
        # TODO - lidar com limpeza de outliers para o sinal com apenas 1 coluna
        var = templates.var(axis=0)

    if way == 'q':
        q3 = np.quantile(var, 0.75)
        q1 = np.quantile(var, 0.25)
        iqr = q3 - q1
        templates = templates[~((var < (q1 - 1.5 * iqr)) | (var > (q3 + 1.5 * iqr)))]

    # elif way == 'z':
    #    new_features = features[(zscore(features) < 3).all(axis=1)]

    # print(len(features)-len(new_features), ' points removed')

    return templates


def normalise_feats(features, norm='minmax'):

    if norm == 'stand':
        return (features - features.mean())/(features.std())

    elif norm == 'minmax':
        return (features - features.min())/(features.max()-features.min())


def correlation_feats(features, th=0.99):

    fcorr = features.corr('pearson') >= th

    idx = np.argwhere(fcorr.values == True)
    corr_idx = idx[np.argwhere(idx[:, 0] != idx[:, 1])]

    idx_ = corr_idx.reshape(-1, 2)[:, 1][:len(corr_idx) // 2]

    return features.drop(columns=features.columns[idx_])


