import numpy as np


def column_index(target_list, query_cols):
    column_idx = []
    for i in np.arange(0, len(query_cols.columns.to_list())):
        temp = target_list.index(str(query_cols.columns.to_list()[i]))
        column_idx.append(temp)
    return column_idx

# a = tp_matrix[0]
# test = pd.DataFrame(0, index=np.arange(len(all_chords)), columns=all_chords)
# test.iloc[column_index(all_chords, a)[1], column_index(all_chords, a)[1]] = a.iloc[1,1]
# for i in np.arange(0, len(a.columns)):
#     for j in np.arange(0, len(a.index)):
#         test.iloc[column_index(all_chords, a)[j], column_index(all_chords, a)[i]] = a.iloc[j,i]
# test = pd.DataFrame(np.matrix(test.iloc[:,:]), index=all_chords, columns=all_chords)



