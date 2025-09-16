from .lunar_area_module import *


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


def N_pmf(
    ds, area, dmin, d_random=1.05, n_samples=3000,
    dmax=None, d_min=None, n_stds=5, sfd_rv=None,
    bin_width_exponent=neukum_bwe, d_max=None, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    p_cutoff=1E-10
):
    if d_max is None:
        d_max = 10 * np.max(ds)
    if dmax is None:
        dmax = d_max
    if sfd_rv is None:
        sfd_rv = sash_pdf(**match_args(locals(), sash_pdf))

    sampled_ds = ds[ds > dmin / np.max(d_random)**n_stds]

    if type(d_random) in {list, np.ndarray}:
        d_rvs = [
            true_d_pdf(d, d_random_i, n_points=10000) 
            for d, d_random_i in zip(sampled_ds, d_random)
        ]
    else:
        d_rvs = [true_d_pdf(d, d_random) for d in sampled_ds]

    def log_interp(sfd_rv, x):
        pos = sfd_rv.P > 0
        logX, logP = np.log10(sfd_rv.X[pos]), np.log10(sfd_rv.P[pos])
        return 10**np.interp(
            np.log10(x), logX, logP, left=logP[0], right=logP[-1]
        )

    ps = []
    for d_rv in d_rvs:
        X, P = d_rv.X, d_rv.P
        w = log_interp(sfd_rv, X) * P * X**2
        p = w[(X >= dmin) & (X < dmax)].sum() / w.sum()
        ps.append(float(np.clip(p, 0.0, 1.0)))
    ps = np.array(ps, dtype=float)
    
    pmf = poisson_binomial_pmf(ps)
    N_array = np.arange(pmf.size)
    pmf = pmf / pmf.sum()
    N_array = N_array[pmf > p_cutoff * pmf.max()]
    pmf = pmf[pmf > p_cutoff * pmf.max()]
    
    return DiscreteRandomVariable(N_array, pmf)


def build_posterior_lookup(sfd_rv, dmin, d_random, n_points=10000):
    s = float(np.log(d_random))
    
    t = np.linspace(
        np.log(sfd_rv.X.min()), 
        np.log(sfd_rv.X.max()), 
        n_points
    )
    X = np.exp(t)

    def log_interp(sfd_rv, x):
        pos = sfd_rv.P > 0
        logX, logP = np.log10(sfd_rv.X[pos]), np.log10(sfd_rv.P[pos])
        return 10**np.interp(
            np.log10(x), logX, logP, left=logP[0], right=logP[-1]
        )
        
    pi = log_interp(sfd_rv, X) * X

    dt = t[1] - t[0]
    half = max(int(np.ceil(4 * s / dt)), 1)
    u = np.arange(-half, half + 1) * dt
    phi = np.exp(-0.5 * (u / s)**2)
    phi /= (phi.sum() * dt)

    def conv_same(a, k):
        pad = (len(k)//2,)
        a_pad = np.pad(a, pad, mode="edge")
        out = np.convolve(a_pad, k, mode="valid")
        return out

    den = conv_same(pi, phi)
    pi_trunc = pi.copy()
    pi_trunc[(t < np.log(dmin)) | (t >= np.log(dmax))] = 0.0
    num = conv_same(pi_trunc, phi)

    tiny = 1e-300
    ratio = num / np.maximum(den, tiny)

    def p_of_d(ds):
        y = np.log(ds) + 0.5 * s*s
        f = np.clip(np.interp(
            y, t, ratio, left=ratio[0], right=ratio[-1]
        ), 0.0, 1.0)
        return f

    return p_of_d


def N_pmf_fast(
    ds, area, dmin, d_random=1.05,
    dmax=None, d_min=None, n_stds=4, sfd_rv=None,
    bin_width_exponent=neukum_bwe, d_max=None,
    growth_rate=1.3, p_cutoff=1E-10, n_points=10000
):
    if d_max is None:
        d_max = 10 * float(np.max(ds))
    if dmax is None:
        dmax = d_max
    if sfd_rv is None:
        sfd_rv = sash_pdf(**match_args(locals(), sash_pdf))

    sampled_ds = ds[ds > dmin / (float(d_random)**n_stds)]
    if sampled_ds.size == 0:
        return DiscreteRandomVariable(np.array([0]), np.array([1.0]))

    p_lookup = build_posterior_lookup(
        sfd_rv, dmin, d_random, n_points=n_points
    )
    ps = p_lookup(sampled_ds)

    pmf = poisson_binomial_pmf(ps)
    N_array = np.arange(pmf.size)
    pmf = pmf / pmf.sum()
    N_array = N_array[pmf > p_cutoff * pmf.max()]
    pmf = pmf[pmf > p_cutoff * pmf.max()]

    return DiscreteRandomVariable(N_array, pmf)


def N1_pdf(
    N, area, dmin, pf=npf_new, kind='median',
    d_systematic=1.0, pf_error=1.0, npl=2000, lambda_rv=None,
    random=1.5, systematic=1.1, additional=1.1, sfd_rv=None
):
    if lambda_rv is None:
        lambda_rv = lambda_pdf(
            **match_args(locals(), lambda_pdf), apply_error=True
        )
    if pf_error is None:
        pf_error = 1.0
    crater_density_rv = lambda_rv / area
    if d_systematic != 1.0:
        if not isinstance(d_systematic, MathRandomVariable):
            d_systematic = apply_factor(1.0, d_systematic)
        if sfd_rv is None:
            raise ValueError(
                'If d_systematic is not 1.0, you must input sfd_rv.'
            )
        sfd_rv2 = sfd_rv.update(sfd_rv)
        pf_dmin = (dmin * d_systematic).update(sfd_rv2).apply(pf)
    elif isinstance(dmin, MathRandomVariable):
        pf_dmin = dmin.apply(pf)
    else:
        pf_dmin = pf(dmin)
    N1_shift = pf(1) / (pf_dmin * pf_error)
    if isinstance(dmin, MathRandomVariable):
        std_ratio = crater_density_rv.std() / lambda_rv.std() * area
        n = int(max(50, 2 * dmin.X.shape[0] * std_ratio))
        crater_density_rv = downsample(crater_density_rv, n)
    N1_rv = crater_density_rv * N1_shift
    return N1_rv


def age_pdf(
    ds, area, dmin, dmax=None, pf=npf_new, cf_inv=ncf_inv, 
    kind='median', d_random=1.05, 
    d_systematic=1.01, pf_error=1.0, npl=2000, cf_error=1.0, 
    random=1.5, systematic=1.1, additional=1.1, sfd_rv=None, 
    d_min=None, bin_width_exponent=neukum_bwe, d_max=None, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    n_N1_points=10000, n_dmin_samples=50
):

    if cf_error is None:
        cf_error = 1.0
    if d_random is None:
        d_random = 1.0
    if d_systematic is None:
        d_systematic = 1.0
    diameter_error = (d_random != 1.0) or (d_systematic != 1.0)
    if diameter_error and (sfd_rv is None):
        args = match_args(locals(), sash_pdf, exclude='kind')
        sfd_rv = sash_pdf(**args)
    
    if isinstance(dmin, MathRandomVariable):
        sampled_rv = dmin.trim(0.99).downsample(n_dmin_samples)
        dmins, dmin_weights = sampled_rv.X, sampled_rv.P
        N1_pdf_kwargs = match_kwargs(locals(), N1_pdf)
        N_pmf_kwargs = match_kwargs(locals(), N_pmf)
        N_rvs = [
            N_pmf(ds, area, _dmin, **N_pmf_kwargs)
            for _dmin in dmins
        ]
        N1_rvs = [
            N1_pdf(N_rv, area, _dmin, **N1_pdf_kwargs)
            for N_rv, _dmin in zip(N_rvs, dmins)
        ]
        N1_min = np.min([N1_rv.X.min() for N1_rv in N1_rvs])
        N1_max = np.max([N1_rv.X.max() for N1_rv in N1_rvs])
        X = np.linspace(N1_min, N1_max, n_N1_points)
        N1_rv_Ps = [N1_rv.match_X(X).P for N1_rv in N1_rvs]
        N1_rv_matrix = np.array([
            P * w for P, w in zip(N1_rv_Ps, dmin_weights)
        ])
        summed_P = N1_rv_matrix.sum(axis=0)
        N1_rv = RandomVariable(X, summed_P, kind=kind)
    else:
        if (d_random is None) or (d_random == 1.0):
            N = ds[ds > dmin].shape[0]
        else:
            N = N_pmf(**match_args(locals(), N_pmf))
        N1_rv = N1_pdf(**match_args(locals(), N1_pdf))
    
    return (N1_rv * cf_error).apply(cf_inv)


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


