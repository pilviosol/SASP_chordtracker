import numpy as np


def column_index(list, query_cols):
    column_idx = []
    for i in np.arange(0, len(query_cols.columns.to_list())):
        temp = list.index(str(query_cols.columns.to_list()[i]))
        column_idx.append(temp)
    return sorted(column_idx)