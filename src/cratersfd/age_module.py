from .pareto_module import *


def split_by_N(dmin_rv, ds):
    ds_in_X = ds[(ds < dmin_rv.X.max()) & (ds > dmin_rv.X.min())]
    N_list = list(np.flip(np.where(np.isin(np.flip(np.sort(ds)), ds_in_X))[0]))
    X, C = dmin_rv.X, dmin_rv.C()
    
    def _slice(rv, a, b):
        r = rv[(rv.X > a) & (rv.X < b)]
        if r.X.shape[0] < 100:
            r = rv.match_X(np.linspace(a, b, 100))
        return r
    
    core_dmin_rv_list = [
        [
            N_list[i], 
            _slice(dmin_rv, ds_in_X[i], ds_in_X[i + 1]).normalize(),
            np.interp(ds_in_X[i + 1], X, C) - np.interp(ds_in_X[i], X, C)
        ]
        for i in range(len(ds_in_X) - 1)
    ]
    first = [
        N_list[0] + 1,
        dmin_rv.match_X(
            np.linspace(dmin_rv.X.min(), ds_in_X[0], 1000), recalculate=True
        ).normalize(),
        np.interp(ds_in_X[0], X, C)
    ]
    last = [
        N_list[-1], 
        dmin_rv.match_X(
            np.linspace(ds_in_X[-1], dmin_rv.X.max(), 1000), recalculate=True
        ).normalize(),
        1 - np.interp(ds_in_X[-1], X, C)
    ]
    return [first] + core_dmin_rv_list + [last]


def lambda_error_lognormal(
    N, random=1.5, systematic=1.1, statistical=1.2
):
    return np.exp(np.sqrt(np.sum([
        np.log(random)**2 / N,
        np.log(systematic)**2,
        np.log(statistical)**2
    ])))


def N1_pdf(
    N, area, dmin, pf=npf_new, kind='median', lambda_error=1.0,
    d_error=1.0, pf_error=1.0, npl=2000, do_trimming=True,
    random=1.5, systematic=1.1, statistical=1.2
):
    dmin_is_an_rv = isinstance(dmin, MathRandomVariable)
    d_error_is_an_rv = isinstance(d_error, MathRandomVariable)
    lambda_error_is_an_rv = isinstance(lambda_error, MathRandomVariable)
    if lambda_error_is_an_rv:
        l_e = downsample(lambda_error, npl)
    elif lambda_error is None:
        l_e = downsample(apply_factor(1, lambda_error_lognormal(
            **match_args(locals(), lambda_error_lognormal)
        )), npl)
    elif lambda_error == 1.0:
        l_e = lambda_error
    else:
        l_e = downsample(apply_factor(1, lambda_error), npl)
    lambda_rv = lambda_pdf(N, kind=kind, n_points=npl) * l_e
    if do_trimming and isinstance(lambda_error, MathRandomVariable):
        lambda_rv = lambda_rv.trim()
    crater_density_rv = lambda_rv / area
    if dmin_is_an_rv or d_error_is_an_rv:
        we = dmin * d_error
        pf_dmin = we.apply(pf)
    else:
        pf_dmin = pf(dmin)
    N1_shift = pf(1) / (pf_dmin * pf_error)
    if dmin_is_an_rv:
        std_ratio = crater_density_rv.std() / lambda_rv.std() * area
        n = int(max(50, 2 * dmin.X.shape[0] * std_ratio))
        crater_density_rv = downsample(crater_density_rv, n)
    N1_rv = crater_density_rv * N1_shift
    return N1_rv


def age_pdf(
    N_or_ds, area, dmin, pf=npf_new, cf_inv=ncf_inv, kind='median',
    lambda_error=None, d_error=1.0, pf_error=1.0, npl=2000, cf_error=1.0,
    random=1.5, systematic=1.1, statistical=1.2
):
    if isinstance(dmin, MathRandomVariable):
        ds = N_or_ds
        dmin_list = split_by_N(dmin, ds)
        N1_rv_list = [
            [li[0], N1_pdf(
                li[0], area, li[1], 
                **match_kwargs(locals(), N1_pdf)
            ), li[2]]
            for li in dmin_list
        ]
        
        X_list = [rvi[1].X for rvi in N1_rv_list]
        Xmin = np.min([Xi.min() for Xi in X_list])
        Xmax = np.max([Xi.max() for Xi in X_list])
        Xlen = np.max([Xi.shape[0] for Xi in X_list])
        N1X = np.linspace(Xmin, Xmax, Xlen, endpoint=True)
        N1_P_array = np.array([
            li[1].match_X(N1X).P * li[2]
            for li in N1_rv_list
        ])

        N1_rv = RandomVariable(
            N1X, np.sum(N1_P_array, axis=0), kind='median'
        )

    elif type(N_or_ds) in {list, np.ndarray}:
        N = N_or_ds[N_or_ds > dmin].shape[0]
        N1_rv = N1_pdf(**match_args(locals(), N1_pdf))
    else:
        N = N_or_ds
        N1_rv = N1_pdf(N, area, dmin, **(match_kwargs(locals(), N1_pdf)))
    
    return (N1_rv * cf_error).apply(cf_inv)


def apply_factor(n, factor, n_stds=6, n_points=None):
    if n_points is None:
        if isinstance(n, MathRandomVariable):
            Xlen = n.X.shape[0]
        else:
            Xlen = 10000
    else:
        Xlen = n_points
    Xmin = factor**(-1 * n_stds)
    Xmax = factor**n_stds
    X = np.linspace(Xmin, Xmax, Xlen)
    P = lognorm.pdf(X, s=np.log(factor), scale=1)
    X /= np.exp(np.log(factor)**2 / 2)
    factor_rv = RandomVariable(X, P, kind='median')
    return n * factor_rv


def m16_age_p(N, A, dmin, pf, cf, t):
    Cdmin = pf(dmin) / pf(1) * cf(t)
    return np.exp(-1 * A * Cdmin) * cf(t)**N


def m16_age_pdf(
    N, A, dmin, pf=npf_new, cf=ncf, cf_inv=ncf_inv, kind='median'
):
    N_min = true_error_pdf(N).percentile(0.00001)
    N_max = true_error_pdf(N).percentile(0.99999)
    T_min = cf_inv(float(N_min / A * pf(1) / pf(dmin)))
    T_max = cf_inv(float(N_max / A * pf(1) / pf(dmin)))
    T = np.linspace(T_min, T_max, 10000)
    P = m16_age_p(N, A, dmin, pf, cf, T)
    return RandomVariable(T, P, kind=kind)


def age_scaled_N1_pdf(
    N, area, dmin, pf=npf_new, cf_inv=ncf_inv, kind='log'
):
    N1_rv = N1_pdf(N, area, dmin, pf=pf, kind=kind)
    return RandomVariable(cf_inv(N1_rv.X), N1_rv.P)


