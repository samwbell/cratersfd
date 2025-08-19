from .pareto_module import *
from .generic_plotting_module import *


def differential_correction(w, m):
    return m * (w**0.5 - w**-0.5) / (w**(m/2) - w**(-1*m/2))


def fast_calc_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None,
    start_at_reference_point=False, growth_rate=1.0,
    fraction=1.0, pf=npf_new_loglog, do_correction=True,
    full_output=False, x_axis_position='linear_center'
):

    if d_min is not None:
        reference_point = d_min
        start_at_reference_point = True

    if d_max is None:
        dmax = 1E4
    else:
        dmax = d_max

    counts, bins, bin_min, bin_max = bin_craters(
        **match_args(locals(), bin_craters)
    )
    
    widths = bins[1:] - bins[:-1]
    differential_counts = counts / widths

    if do_correction:
        rise = pf(np.log10(bins[1:])) - pf(np.log10(bins[:-1]))
        run = np.log10(bins[1:]) - np.log10(bins[:-1])
        m = rise / run
        w = 2**bin_width_exponent
        cfs = differential_correction(w, m)
    else:
        cfs = np.ones(counts.shape[0])
        
    differential_counts *= cfs
    
    diameter_array = get_bin_parameters(
        **match_args(locals(), get_bin_parameters)
    )
    
    sorted_ds = diameter_array
    densities = differential_counts / area
    
    if full_output:
        return sorted_ds, densities, counts, widths, bins, cfs
    else:
        return sorted_ds, densities
fast_calc_dif = fast_calc_differential
    

def calc_differential_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None, growth_rate=1.0,
    fraction=1.0, start_at_reference_point=False, pf=npf_new_loglog,
    do_correction=True, x_axis_position='linear_center', kind='log'
):
    sorted_ds, rhos, counts, widths, bins, cfs = fast_calc_differential(
        **match_args(locals(), fast_calc_differential), full_output=True
    )
    lambda_pdfs = lambda_pdf(counts, kind=kind)
    density_pdf_list = cfs / widths / area * lambda_pdfs
    return sorted_ds, density_pdf_list, bins
calc_dif_pdfs = calc_differential_pdfs


def plot_differential(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None,
    start_at_reference_point=False, growth_rate=1.0,
    fraction=1.0, pf=npf_new_loglog, 
    do_correction=False, ax='None', color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True,
    x_axis_position='linear_center', ms=4, kind='log'
):
    args = match_args(locals(), calc_differential_pdfs)
    bin_ds, pdf_list, bins = calc_differential_pdfs(**args)
    
    args = match_args(locals(), plot_pdf_list)
    args.pop('ds')
    plot_pdf_list(bin_ds, **args, ylabel_type = 'Differential')
    
    return np.array(bins)
plot_dif = plot_differential


def plot_dif_N(
    N, area, pf=npf_new_dif, dmin=0.005, dmax=1E3, color='k', lw=1
):
    Xpf = np.logspace(np.log10(dmin), np.log10(dmax), 10000)
    dif_integral = trapezoid(pf(Xpf), Xpf)
    Ypf = pf(Xpf) / dif_integral * N / area
    plt.plot(Xpf, Ypf, color, lw=lw)
plot_npf_dif_N = plot_dif_N


def dif_line(
    alphas, lambdas, area, bins, n_points=10000, 
    X=None, return_X=False
):
    if X is None:
        X = np.logspace(
            np.log10(bins[0]), np.log10(bins[-1]), n_points
        )
        X[0] = bins[0]
        X[-1] = bins[-1]
        return_X = True
    else:
        return_X = False
    Y = np.array([])
    n_bins = bins.shape[0] - 1
    for i in range(n_bins):
        ei, ei1, ai = bins[i], bins[i + 1], alphas[i]
        if i < n_bins - 1:
            Xi = X[(X >= ei) & (X < ei1)]
        else:
            Xi = X[(X >= ei) & (X <= ei1)]
        raw_Y = tp_eq(Xi, dmin=ei, dmax=ei1, alpha=ai) 
        Yi = raw_Y * lambdas[i] / area
        Y = np.append(Y, Yi)
    if return_X:
        return X, Y
    else:
        return Y
        

def dif_line_from_ds(
    ds, area, bins, n_points=10000, X=None, n_alpha_points=10000,
    return_alphas=False
):
    if X is None:
        X = np.logspace(
            np.log10(bins[0]), np.log10(bins[-1]), n_points
        )
        X[0] = bins[0]
        X[-1] = bins[-1]
    counts, bin_array = np.histogram(ds, bins)
    tp_rvs = [
        truncated_pareto_pdf(
            ds, dmin=bins[i], dmax=bins[i + 1],
            n_points=n_alpha_points
        ).as_kind('mean')
        for i in range(bins.shape[0] - 1)
    ]
    tp_alphas = [rv.mean() for rv in tp_rvs]
    if return_alphas:
        return dif_line(
            tp_alphas, counts, area, bins, n_points=n_points, X=X
        ), tp_alphas
    else:
        return dif_line(
            tp_alphas, counts, area, bins, n_points=n_points, X=X
        )   


def solve_dif_hist(
    ds, bins, smoothing_strength=1.0,
    alpha_min=0.3, alpha_max=10, 
    lambda_min=1e-3, lambda_max=None,
    do_printing=False
):

    counts, _ = np.histogram(ds, bins)
    n_bins = len(counts)
    
    def truncated_pareto_loglike(D, alpha, Dmin, Dmax):
        if alpha == 1:
            norm = np.log(Dmax / Dmin)
        else:
            norm = (Dmax**(1 - alpha) - Dmin**(1 - alpha)) / (1 - alpha)
        return np.sum(-alpha * np.log(D) - np.log(norm))
    
    def neg_log_posterior(params):
        alphas = params[:n_bins]
        lambdas = params[n_bins:]
        loglike = 0
        n = 0
    
        for i in range(n_bins):
            Dmin, Dmax = bins[max(0, i - n)], bins[min(n_bins, i + n + 1)]
            D_i = ds[(ds >= Dmin) & (ds < Dmax)]
            k_i = counts[i]
            λ_i = lambdas[i]
            α_i = alphas[i]
            loglike += poisson.logpmf(k_i, λ_i)
            if len(D_i) > 0:
                loglike += truncated_pareto_loglike(D_i, α_i, Dmin, Dmax)
    
        # return -loglike
    
        smoothing_penalty = 0
        for i in range(n_bins - 1):
            frac_diff = (alphas[i + 1] - alphas[i]) / alphas[i]
            min_count = min(counts[i], counts[i + 1])
            weight = smoothing_strength / (1 + min_count)
            smoothing_penalty += weight * frac_diff**2
            
        return -loglike + smoothing_penalty
    
    def continuity_constraint(params):
        alphas = params[:n_bins]
        lambdas = params[n_bins:]
        cons = []
    
        for i in range(n_bins - 1):
            e_i, e_ip1, e_ip2 = bins[i], bins[i+1], bins[i+2]
            α_i, α_ip1 = alphas[i], alphas[i+1]
            λ_i, λ_ip1 = lambdas[i], lambdas[i+1]
    
            term_num = (e_ip1 / e_i)**α_i - 1
            term_den = 1 - (e_ip1 / e_ip2)**α_ip1
    
            lhs = λ_i / λ_ip1
            rhs = (α_ip1 / α_i) * (term_num / term_den)
    
            cons.append(lhs - rhs)
    
        return np.array(cons)
    
    init_alphas = np.full(n_bins, 2.0)
    init_lambdas = np.maximum(counts, 1.0)
    x0 = np.concatenate([init_alphas, init_lambdas])

    alpha_bounds = [(alpha_min, alpha_max)] * n_bins
    lambda_bounds = [(lambda_min, lambda_max)] * n_bins
    bounds = alpha_bounds + lambda_bounds
    
    result = minimize(
        neg_log_posterior,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': continuity_constraint},
        options={'ftol': 1e-8, 'maxiter': 1000, 'disp': do_printing}
    )

    if result.success:
        opt_alphas = result.x[:n_bins]
        opt_lambdas = result.x[n_bins:]
        if do_printing:
            print("Optimal alphas:", opt_alphas)
            print("Optimal lambdas:", opt_lambdas)
    else:
        print("Optimization failed:", result.message)

    return opt_alphas, opt_lambdas


def digitize_bins(X, bins, values):
    indices = np.digitize(X, bins) - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices.astype(int)]


def calc_sash(
    ds, area, d_min=None, 
    bin_width_exponent=neukum_bwe, d_max=None, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    return_alphas=False
):
    
    fractions = 1.0 - np.linspace(0, 1, n_shifts + 1, endpoint=False)

    if (d_min is None) or (d_min == np.min(ds)):
        d_min = safe_d_min(ds)

    if d_max is None:
        d_max = 1.3 * np.max(ds)

    bin_info_list = [
        bin_craters(
            **match_args(locals(), bin_craters),
            start_at_reference_point=True, reference_point=d_min
        )
        for fraction in fractions
    ]
    counts_list, bins_list, _, _ = zip(*bin_info_list)
    
    X = np.logspace(np.log10(d_min), np.log10(d_max), n_points)
    X[0] = d_min
    X[-1] = d_max

    dif_line_params = [
        dif_line_from_ds(
            ds, area, bins, X=X, n_alpha_points=n_alpha_points,
            n_points=n_points, return_alphas=True
        ) 
        for bins in bins_list
    ]

    Ys, alphas_list = zip(*dif_line_params)

    mean_Y = np.mean(Ys, axis=0)

    tp_alpha_matrix = [
        np.interp(X, bins[:-1], alphas)
        for bins, alphas in zip(bins_list, alphas_list)
    ]
    mean_tp_Alpha = np.mean(tp_alpha_matrix, axis=0)

    for i in range(n_iterations - 1):
        
        alphas_list = []
        for bins in bins_list:
            alphas = []
            for i in range(bins.shape[0] - 1):
                ei, ei1 = bins[i], bins[i + 1]
                ei1 = min(ei1, np.max(ds) * 3)
                logYi = np.log10(np.interp(ei, X, mean_Y))
                logYi1 = np.log10(np.interp(ei1, X, mean_Y))
                rise = logYi1 - logYi
                run = np.log10(ei1 / ei)
                alphas.append(-1 * rise / run - 1)
            alphas_list.append(alphas)
            
        Ys = [
            dif_line(
                alphas, counts, area, bins, X=X,
                n_points=n_points
            ) 
            for counts, bins, alphas 
            in zip(counts_list, bins_list, alphas_list)
        ]
        
        mean_Y = np.mean(Ys, axis=0)

    if return_alphas:
        alpha_matrix = [
            np.interp(X, bins[:-1], alphas)
            for bins, alphas in zip(bins_list, alphas_list)
        ]
        mean_Alpha = np.mean(alpha_matrix, axis=0)
        return (
            X, mean_Y, Ys, mean_Alpha, alpha_matrix,
            mean_tp_Alpha, tp_alpha_matrix
        )
    else:
        return X, mean_Y, Ys
calc_ash_dif = calc_sash


def plot_sash(
    ds, area, d_min=None, 
    bin_width_exponent=per_decade(18), d_max=None, 
    growth_rate=1.2, n_points=10000, n_shifts=200,
    color='mediumslateblue', plot_lines=False, lw=1.2,
    line_color='mediumslateblue', line_lw=0.2,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    return_alphas=False, plot_error=True,
    fill_alpha=0.15, kernel=None, reduction_factor=1.0,
    error_bin_width_exponent=per_decade(18),
    error_downsample=10, kind='log'
):

    t1 = time.time()
    result_tuple = calc_sash(**match_args(locals(), calc_sash))
    if return_alphas:
        (
            X, mean_Y, Ys, mean_Alpha, alpha_matrix,
            mean_tp_Alpha, tp_alpha_matrix
        ) = result_tuple
    else:
        X, mean_Y, Ys = result_tuple
    t2 = time.time()
    print(format_runtime(t2 - t1))

    if plot_lines:
        for Y in Ys:
            plt.plot(X, Y, color=line_color, lw=line_lw)

    plt.plot(X, mean_Y, color, lw=lw)

    Ycut = mean_Y[X <= 100 * np.max(ds)]
    Xcut = X[X <= 100 * np.max(ds)]

    if plot_error:
        Xp = Xcut[::error_downsample]
        Yp = Ycut[::error_downsample]
        C = cumulative_trapezoid(Yp, Xp, initial=0) * area
        bwe = error_bin_width_exponent
        left_edges = np.maximum(Xp / 2**(bwe / 2), Xp[0])
        right_edges = np.minimum(Xp * 2**(bwe / 2), Xp[-1])
        Ns = np.interp(right_edges, Xp, C) - np.interp(left_edges, Xp, C)
        if kernel is not None:
            lambda_rvs = [
                log__mul__(lambda_pdf(
                    N / reduction_factor
                ).as_kind('mean'), kernel) 
                for N in Ns
            ]
        else:
            lambda_rvs = [lambda_pdf(N / reduction_factor) for N in Ns]
        rho_rvs = [
            lambda_rv.as_kind(kind) / lambda_rv.mode() * Y
            for lambda_rv, N, Y in zip(lambda_rvs, Ns, Yp)
        ]
        
        val = np.array([rv.val for rv in rho_rvs])
        low = np.array([rv.low for rv in rho_rvs])
        high = np.array([rv.high for rv in rho_rvs])

        plt.fill_between(
            Xp, low, high, facecolor=color, alpha=fill_alpha
        )

        format_cc_plot(
            Xp, val, low, high, ylabel_type='Differential ',
            x_max_pad=0
        )

    else:

        format_cc_plot(
            Xcut, Ycut, Ycut, Ycut, ylabel_type='Differential ',
            x_max_pad=0
        )

    if return_alphas:
        return X, mean_Y, mean_Alpha, mean_tp_Alpha
    else:
        return X, mean_Y
plot_ash_dif = plot_sash


def calc_differential_binned(
    ds, area, bin_width_exponent=neukum_bwe,
    d_min=None, d_max=1E4, growth_rate=1.0,
    fraction=1.0, min_count=1,
    bins=None, plotting_n=30, alpha=None,
    kind='mean'
):
    if (d_min is None) and (bins is None):
        d_min = safe_d_min(ds)
    if bins is not None:
        counts, _ = np.histogram(ds, bins)
    else:
        counts, bins, bin_min, bin_max = bin_craters(
            **match_args(locals(), bin_craters),
            start_at_reference_point=True
        )
        
    tp_rvs = [
        truncated_pareto_pdf(
            ds, dmin=bins[i], dmax=bins[i + 1]
        ).as_kind('mean')
        for i in range(bins.shape[0] - 1)
    ]
    
    if alpha is None:
        alpha = np.linspace(1E-5, 10, 10000)
    rho_rvs = []
    d_list = []
    for i in range(len(bins) - 1):
        ei, ei1, ai = bins[i], bins[i + 1], tp_rvs[i]
        plot_ds = np.logspace(
            np.log10(ei + 1e-5),
            np.log10(min(ei1, 5 * ds.max())),
            plotting_n
        )
        rhoi = (lambda_pdf(counts[i]) / area)
        ai_rhos = [
            log__mul__(
                apply2rv(
                    ai, lambda x : tp_eq(d, dmin=ei, dmax=ei1, alpha=x), 
                    even_out=False
                ),
                rhoi
            ).as_kind(kind)
            for d in plot_ds
        ]
        rho_rvs += ai_rhos
        d_list += list(plot_ds)

    return np.array(d_list), rho_rvs, bins


def plot_differential_binned(
    ds, area, bin_width_exponent=neukum_bwe,
    d_min=None, d_max=1E4, growth_rate=1.0,
    fraction=1.0, min_count=1, bins=None,
    kind='log', color='mediumslateblue', lw=1.5,
    fill_alpha=0.15, alpha=None, plotting_n=30
):
    point_ds, rho_rvs, bins = calc_differential_binned(
        **match_args(locals(), calc_differential_binned)
    )

    val = np.array([rv.val for rv in rho_rvs])
    low = np.array([rv.low for rv in rho_rvs])
    high = np.array([rv.high for rv in rho_rvs])
    
    plt.plot(point_ds, val, color=color, lw=lw)
    plt.fill_between(
        point_ds, low, high, facecolor=color, alpha=fill_alpha
    )
    
    format_cc_plot(
        point_ds, val, low, high, ylabel_type='Differential ',
        x_max_pad=0
    )


def plot_kde(
    ds, area, d_min=None, d_max=1E4, n_points=10000,
    color='black', lw=1.5
): 
    if d_min is None:
        d_min = np.min(ds)
    logD = np.linspace(np.log10(d_min), np.log10(d_max), n_points)
    D = 10**logD
    ds = np.flip(np.sort(ds))
    i = 0
    log_ds = np.log10(ds[:,np.newaxis])
    kde_matrix = norm.pdf(logD, log_ds, np.log10(1.1)) / area
    normalization = 1 / ds[:,np.newaxis] / math.log(10)
    kde = np.sum(kde_matrix * normalization, axis=0)
    plt.plot(D, kde, color=color, lw=lw)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([d_min, 2 * np.max(ds)])
    in_range = (D >= d_min) & (D <= 2 * np.max(ds))
    plt.ylim([kde[in_range].min(), kde[in_range].max()])


def plot_sash_alpha(
    X, Y, window_length=500, polyorder=5, deriv=1,
    color='mediumslateblue'
):
    logX = np.log10(X)
    logY = np.log10(Y)
    dYdX = savgol_filter(
        logY, window_length=window_length, 
        polyorder=polyorder, deriv=deriv, 
        delta=logX[1] - logX[0]
    )
    plt.plot(X, -1 * dYdX - 1, color=color)
    plt.xscale('log')
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Crater Diameter (km)', size=12)
    plt.ylabel(rf'Negative Cumulative Slope $\alpha$', size=12)
ash_alpha_plot = plot_sash_alpha


def ash_synth(
    X, mean_Y, area, dmin, n_synths=100,
    bin_width_exponent=neukum_bwe, dmax=1E4, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000
):
    def ash_fit_pf(X, mean_Y):
        def f(D):
            return np.interp(D, X, mean_Y)
        return f
    
    synth_ash_fit_d_list = synth_fixed_N(
        N=np.array(ds).shape[0], dmin=dmin, 
        differential_pf=ash_fit_pf(X, mean_Y),
        n_steps=n_synths
    )
    
    synth_mean_Ys = []
    for synth_ds in synth_ash_fit_d_list[0]:
        synth_X, synth_mean_Y, Ys = calc_ash_dif(
            synth_ds, area, dmin,
            bin_width_exponent=bin_width_exponent,
            growth_rate=growth_rate, n_points=n_points,
            n_shifts=n_shifts, min_count=min_count,
            n_iterations=n_iterations,
            n_alpha_points=n_alpha_points
        )
        synth_mean_Ys.append(synth_mean_Y)

    return np.array(synth_mean_Ys)


