
* augment failures via simulation

* compare to brisque on 2d pngs

* augment scale shear

* check c scores vs d

* determine loss eg cce + spearman across samples ...

from scipy.stats import spearmanr
def compute_spearmanr(y, y_pred):
    spearsum = 0
    cnt = 0
    for col in range(y_pred.shape[1]):
        v = spearmanr(y_pred[:,col], y[:,col]).correlation
        if np.isnan(v):
            continue
        spearsum += v
        cnt += 1
    res = spearsum / cnt
    return res

a = np.array([[2., 1., 2., 3.],[3., 3., 4., 5.]] )
b = np.array([[1., 0., 0., 3.], [1., 0., 3., 3.]])
compute_spearmanr(a, b)
