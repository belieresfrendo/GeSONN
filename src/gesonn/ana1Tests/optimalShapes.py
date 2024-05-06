def translate_to_zero(x, y, n_pts):
    xG = x.sum() / n_pts
    yG = y.sum() / n_pts
    return x - xG, y - yG
