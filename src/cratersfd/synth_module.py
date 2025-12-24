from .plot_fitting_module import *
from .cumulative_unbinned_module import *
from .cumulative_binned_module import *
from .differential_module import *
from .R_module import *
from .slope_plot_module import *

def randomize_synth_data(
    synth_list_raw, left_edge_lambda, right_edge_lambda, inc
):
    synth_list_raw_flattened = np.array([
        item for row in synth_list_raw for item in row
    ])
    mean_lambda_drop = np.mean(left_edge_lambda / right_edge_lambda)
    shifts = np.linspace(-0.5, 0.5, 1001)
    p_shift = mean_lambda_drop + (1 - mean_lambda_drop) * (shifts + 0.5)
    p_shift = p_shift / p_shift.sum()
    shift_array = np.random.choice(
        shifts, synth_list_raw_flattened.shape[0], p=p_shift
    )
    synth_list_flattened = 10**(
        np.log10(synth_list_raw_flattened) - shift_array * inc
    )
    splitting_indices = np.array([
        ds.shape[0] for ds in synth_list_raw
    ]).cumsum()
    return np.split(synth_list_flattened, splitting_indices)[:-1]


bin_count_dict = {}


def sample_by_Poisson(lambda_array, n_steps, saved_data_on, runtime_off):

    max_count = int(poisson.ppf(0.9999999999, lambda_array[0]))
    N_array = np.arange(0, max_count + 1)

    if saved_data_on:
        log_lambda_bins = np.round(np.log10(lambda_array), 4)
        lambda_keys = [
            tuple([max_count, n_steps, lambda_bin])
            for lambda_bin in log_lambda_bins
        ]
        lambda_saved = np.array([
            lambda_key in bin_count_dict for lambda_key in lambda_keys
        ])
        bin_count_matrix = np.zeros(
            tuple([lambda_array.shape[0], n_steps]), dtype=np.int8
        )
        if not runtime_off:
            n_bins_saved = np.where(lambda_saved)[0].shape[0]
            print(str(n_bins_saved) + ' bins taken from saved data')
        if np.where(lambda_saved)[0].shape[0] != 0:
            saved_keys = [lambda_keys[i] for i in np.where(lambda_saved)[0]]
            bin_count_matrix[np.where(lambda_saved)] = np.array([
                bin_count_dict[lambda_key] for lambda_key in saved_keys
            ])
        if np.where(~lambda_saved)[0].shape[0] != 0:
            unsaved_lambdas = lambda_array[np.where(~lambda_saved)]
            p_array = np.array([
                poisson.pmf(N, unsaved_lambdas) for N in N_array
            ]).T
            bin_count_matrix[np.where(~lambda_saved)] = np.array([
                np.random.choice(N_array, n_steps, p=p) for p in p_array
            ])
        for i in np.where(~lambda_saved)[0]:
            rounded_lambda = np.round(np.log10(lambda_array[i]), 4)
            dict_saving_key = tuple([max_count, n_steps, rounded_lambda])
            bin_count_dict[dict_saving_key] = bin_count_matrix[i]
            
    else:
        p_array = np.array([poisson.pmf(N, lambda_array) for N in N_array]).T
        bin_count_matrix = np.array([
            np.random.choice(N_array, n_steps, p=p) for p in p_array
        ])
        
    return bin_count_matrix.T


def synth_data(
    model_lambda=20, area=10000, pf=loglog_linear_pf(N1=0.001, slope=-2),
    dmin=1, dmax=1E5, dmax_tolerance=0.00001, n_steps=100000, inc=0.001, 
    runtime_off=False, saved_data_on=False, N_must_match_lambda=False
):
    
    t1 = time.time()
    
    synth_age = model_lambda / (10**pf(np.log10(dmin)) * area)
    logd_array = np.arange(
        np.log10(dmin) + inc / 2, np.log10(dmax) + inc / 2, inc
    )
    cumulative_lambda_array = 10**pf(logd_array) * area * synth_age
    logd_array = logd_array[cumulative_lambda_array > dmax_tolerance]
    if not runtime_off:
        print(str(logd_array.shape[0]) + ' lambda bins')
    left_edge_lambda = 10**pf(logd_array - inc/2)
    right_edge_lambda = 10**pf(logd_array + inc/2)
    lambda_array = (left_edge_lambda - right_edge_lambda) * area * synth_age
    slope_array = (left_edge_lambda - right_edge_lambda) / inc
    lambda_array *= differential_correction(10**inc, slope_array)
    bin_count_array = sample_by_Poisson(
        lambda_array, n_steps, saved_data_on, runtime_off
    )
    if N_must_match_lambda:
        bin_count_list = list(
            bin_count_array[bin_count_array.sum(axis=1) == model_lambda]
        )
    else:
        bin_count_list = list(bin_count_array)
    synth_list_raw = [
        10**np.repeat(logd_array, bin_count_array) 
        for bin_count_array in bin_count_list
    ]
    synth_list = randomize_synth_data(
        synth_list_raw, left_edge_lambda, right_edge_lambda, inc
    )
    
    t2 = time.time()
    
    if not runtime_off:
        print('runtime: ' + format_runtime(t2 - t1))
        
    return synth_list, synth_age


def synth_fixed_N(
    N=20, dmin=1, dmax=10**3.5, n_points=10000, 
    pf=loglog_linear_pf(N1=0.001, slope=-2),
    loglog_cumulative_pf=None,
    cumulative_pf=None, differential_pf=None,
    loglog_differential_pf=None, n_steps=100, area=10000
):
    if loglog_cumulative_pf is not None:
        pf = loglog_cumulative_pf
    elif cumulative_pf is not None:
        pf = linear2loglog_pf(cumulative_pf)
    elif differential_pf is not None:
        pf = linear2loglog_pf(
            differential2cumulative_pf(differential_pf)
        )
    elif loglog_differential_pf is not None:
        pf = linear2loglog_pf(
            differential2cumulative_pf(
                loglog2linear_pf(loglog_differential_pf)
            )
        )
    logD = np.flip(np.linspace(np.log10(dmin), np.log10(dmax), n_points))
    D = 10**logD
    Y = 10**pf(logD) - 10**pf(np.log10(dmax))
    P_cumulative = Y / Y.max()
    synth_list = [
        np.interp(np.random.random(N), P_cumulative, D)
        for i in range(n_steps)
    ]
    synth_age = N / (10**pf(np.log10(dmin)) * area)
    return synth_list, synth_age


def pick_a_side_fit(
    sorted_ds, density_array, uncertainties, m_guess, 
    b_guess, lower, upper
):
    m, b = fit_linear(sorted_ds, density_array, 
                      uncertainties=uncertainties, 
                      guess=[m_guess, b_guess])
    continue_iteration = True
    iteration_count = 0
    switch_count = 5
    while continue_iteration and (iteration_count < 5):
        adjusted_uncertainties = uncertainties.copy()
        above_data = 10**(m * np.log10(sorted_ds) + b) > density_array
        adjusted_uncertainties[above_data] = upper[np.where(above_data)]
        adjusted_uncertainties[~above_data] = lower[np.where(~above_data)]
        flipQ = (np.sum(uncertainties != adjusted_uncertainties) > 0)
        if flipQ or (iteration_count == 0):
            m, b = fit_linear(sorted_ds, density_array, 
                              uncertainties = adjusted_uncertainties,
                              guess=[m_guess, b_guess])
        if not flipQ:
            continue_iteration = False
            switch_count = iteration_count
        uncertainties = adjusted_uncertainties.copy()
        iteration_count += 1
    return m, b, switch_count


def get_slope_cumulative_unbinned(
    ds, area, age, uncertainties='asymmetric', d_min=None,
    do_correction=True, warning_off=False, kind='log',
    pf=loglog_linear_pf(N1=0.001, slope=-2)
):
    D, Rho, N = fast_calc_cumulative_unbinned(
        ds, area, return_N=True, kind=kind
    )
    if do_correction:
        D = center_cumulative_points(D, d_min=d_min)
        if d_min is None:
            Rho = Rho[:-1]
            if not warning_off:
                print(
                    'The d_min parameter is currently set to None.  This '
                    'is not recommended.  You should choose a value.  It '
                    'describes the diameter where you began counting.  '
                    'For instance, if you counted craters larger than '
                    '1.0km, then set d_min=1.0.  To suppress this '
                    'warning, set warning_off=True.'
                )
    if uncertainties is None:
        sigma = None
    elif uncertainties in ['asymmetric', 'symmetric']:
        lower, upper = get_error_bars(N, log_space=True, kind=kind)
        sigma = (lower + upper) / 2
    m_guess = pf(1) - pf(0)
    b_guess = pf(0) + np.log10(age)
    if uncertainties=='asymmetric':
        m, b, switch_count = pick_a_side_fit(
            D, Rho, sigma, m_guess, b_guess, lower, upper
        )
    else:
        m, b = fit_linear(
            D, Rho, uncertainties=sigma, guess=[m_guess, b_guess]
        )
        
    return m, b


def get_slope_cumulative_binned(
    ds, area, age, uncertainties='asymmetric', 
    pf=loglog_linear_pf(N1=0.001, slope=-2), 
    bin_width_exponent=neukum_bwe, x_axis_position='left',
    reference_point=1.0, skip_zero_crater_bins=False,
    start_at_reference_point=False, d_max=None
):
    D, Rho, N = fast_calc_cumulative_binned(
        ds, area, bin_width_exponent=neukum_bwe, 
        x_axis_position=x_axis_position,
        reference_point=reference_point, 
        skip_zero_crater_bins=skip_zero_crater_bins,
        start_at_reference_point=start_at_reference_point, 
        d_max=d_max, return_N=True
    )
    if uncertainties is None:
        sigma = None
    elif uncertainties in ['asymmetric', 'symmetric']:
        lower, upper = get_true_error_bars_log_space(N)
        sigma = (lower + upper) / 2
    m_guess = pf(1) - pf(0)
    b_guess = pf(0) + np.log10(age)
    if uncertainties=='asymmetric':
        m, b, switch_count = pick_a_side_fit(
            D, Rho, sigma, m_guess, b_guess, lower, upper
        )
    else:
        m, b = fit_linear(
            D, Rho, uncertainties=sigma, guess=[m_guess, b_guess]
        )
        
    return m, b


def model_fitting_error(
    synth_list, synth_age, synth_area, 
    pf=loglog_linear_pf(N1=0.001, slope=-2),
    bin_width_exponent=neukum_bwe, 
    use_uncertainties=False, kind='log', 
    pick_a_side=False, plot_type='unbinned', 
    d_min=None, skip_zero_crater_bins=False, 
    n_pseudosteps=100, reference_point=1.0, 
    start_at_reference_point=False, 
    print_failures=True
):
    slope_list = []
    failure_list = []
    failure_N_list = []
    failure_reason_list = []
    switch_list = []
    for i in range(n_pseudosteps):
        try:
            synth_ds = synth_list[i]
            
            if synth_ds.shape[0] == 0:
                raise ValueError(
                    'There are no craters in this '
                    'synthetic observation, so '
                    'slope cannot be calculated.'
                )

            if plot_type in {'unbinned', 'unbinned corrected'}:
                sorted_ds, density, lower, upper = fast_calc_cumulative_unbinned(
                    synth_ds, synth_area, calculate_uncertainties=True, 
                    log_space=True, kind=kind
                )
                if plot_type == 'unbinned corrected':
                    sorted_ds = center_cumulative_points(
                        sorted_ds, d_min=d_min
                    )
                    if d_min is None:
                        density = density[:-1]
            else:
                sorted_ds, density = fast_calc_cumulative_binned(
                    synth_ds, synth_area, 
                    bin_width_exponent=neukum_bwe, 
                    x_axis_position=plot_type,
                    skip_zero_crater_bins=skip_zero_crater_bins, 
                    reference_point=reference_point, 
                    start_at_reference_point=start_at_reference_point
                )
                if sorted_ds.shape[0] == 1:
                    raise ValueError(
                        'These craters fall into only '
                        'one bin, so no slope can be fit.'
                    )
                lower, upper = get_error_bars(
                    np.round(density * synth_area, 7), kind=kind,
                    log_space=True
                )

            if use_uncertainties:
                uncertainties = (upper + lower) / 2.0
            else:
                uncertainties = None

            m_guess = pf(1) - pf(0)
            b_guess = pf(0) + np.log10(synth_age)
            if pick_a_side and use_uncertainties:
                m, b, switch_count = pick_a_side_fit(
                    sorted_ds, density, uncertainties, 
                    m_guess, b_guess, lower, upper
                )
                switch_list.append(switch_count)
            else:
                m, b = fit_linear(
                    sorted_ds, density, uncertainties=uncertainties,
                    guess=[m_guess, b_guess]
                )
                
            slope_list.append(m)
            
        except Exception as failure_reason:
            
            failure_list.append(i)
            failure_N_list.append(len(synth_ds))
            failure_reason_list.append(str(failure_reason))
            
    failure_df = pd.DataFrame({'i': failure_list, 'N': failure_N_list, 
                               'Reason': failure_reason_list})        
    if print_failures:
        if len(failure_list) > 0:
            print(failure_df)
            
    if pick_a_side and use_uncertainties:
        return slope_list, switch_list, failure_df
    else:
        return slope_list, failure_df
    
    
def plot_result_pdf(
    data_list, ax=None, label_text_size=10, xlim=None, 
    right_position=0.85, custom_label_height=1.12, 
    label_shift_x=0, reference_value=1.0, 
    n_bins_baseline=50, slope_data=False, upshift=0
):
    if ax is None:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)

    data_dist = np.array(data_list) / reference_value
    data_pdf = make_pdf_from_samples(
        data_dist, slope_data=slope_data, n_bins_baseline=n_bins_baseline
    )
    if xlim is None:
        min_d = data_pdf.X.min()
        max_d = data_pdf.X.max()
    else:
        min_d = xlim[0]
        max_d = xlim[1]
    ax, text_x = data_pdf.plot(
        label='median', return_text_x=True, 
        label_shift_x=label_shift_x,
        label_text_size=label_text_size, 
        force_label_side='left', xlim=xlim,  
        label_shift_y=0.2, upshift=upshift,
        error_bar_type='median'
    )
    text_x = text_x + label_shift_x
    mean_text = "{:.2f}".format(round(np.mean(data_dist[~np.isnan(data_dist)]), 2))
    plt.text(min_d + right_position * (max_d - min_d), custom_label_height, 
             'mean:\n' + mean_text, ha='left', va='center', size=label_text_size)
    plt.text(text_x, custom_label_height, 'median:\n', ha='left', va='center', 
             size=label_text_size)
    
    return ax

def fit_slope_data(slope_list, nbins=50):
    data_pdf = ash_pdf(slope_list, nbins=nbins)
    return fit_slope_pdf(data_pdf.X, data_pdf.P)

def mean_slope_fit(slope_list, nbins=50):
    slope_rv = ash_pdf(slope_list, nbins=nbins)
    return slope_rv.val, slope_rv.lower, slope_rv.upper

def old_fit_slope_data(
    slope_list, n_bins_baseline=50, reference_value=1.0
):
    
    data_dist = np.array(slope_list) / reference_value
    data_pdf = make_pdf_from_samples(
        data_dist, slope_data=True, n_bins_baseline=n_bins_baseline
    )
    
    return fit_slope_pdf(data_pdf.X, data_pdf.P)


def fit_age_data(age_list, n_bins_baseline=50, reference_value=1.0):
    
    data_dist = np.array(age_list) / reference_value
    data_pdf = make_pdf_from_samples(
        data_dist, n_bins_baseline=n_bins_baseline
    )
    
    return fit_log_of_normal(data_pdf.X, data_pdf.P)


def plot_log_fit(slope_list, label_text='', upshift=0, ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        
    slope_pdf = make_pdf_from_samples(slope_list, slope_data=True)
    slope_pdf.flip().log().plot(
        color='mediumslateblue', label=True, rounding_n=3, 
        upshift=upshift, label_color='black'
    )
    log_max, log_lower, log_upper = fit_slope_data(slope_list)
    plot_log_of_normal_fit(
        log_max, log_lower, log_upper, 
        color='black', upshift=upshift
    )
    plt.text(0.7, 0.06 + 1.1, label_text, size=7.5, 
         color='black', ha='right')
    
    
synth_sash_dict = {}
def synth_sash(
    ds, area, d_min=None, use_saved_data=True,
    differential_pf=None, n_steps=100,
    bin_width_exponent=neukum_bwe, d_max=1E4, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000
):

    args = match_args(
        locals(), synth_sash, exclude=['ds', 'use_saved_data']
    )
    args_key = tuple(list(ds) + list(args.values()))

    if use_saved_data and (args_key in synth_sash_dict):

            returns = synth_sash_dict[args_key]

    else:

        if differential_pf is None:
            X, Y = calc_sash(**match_args(locals(), calc_sash))
            differential_pf = fit_pf(X, Y)
        
        synth_d_list = synth_fixed_N(
            N=np.array(ds).size, dmin=safe_d_min(ds), dmax=d_max,
            **match_kwargs(locals(), synth_fixed_N)
        )
    
        synth_mean_Ys = []
        for synth_ds in synth_d_list[0]:
            synth_X, synth_mean_Y, Ys = calc_sash(
                synth_ds, area,
                **match_kwargs(locals(), calc_sash)
            )
            synth_mean_Ys.append(synth_mean_Y)

        returns = synth_X, np.array(synth_mean_Ys)
        synth_sash_dict[args_key] = returns

    return returns


def plot_sash_synth(
    ds, area, d_min=None, plot_type='differential',
    differential_pf=None, n_steps=100,
    bin_width_exponent=neukum_bwe, d_max=1E4, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    color='mediumslateblue', fill_alpha=0.15, lw=1.5, 
    ls=':', fontsize=14, X=None, synth_mean_Ys=None
):
    if (X is None) or (synth_mean_Ys is None):
        X, synth_mean_Ys = synth_sash(
            **match_args(locals(), synth_sash)
        )

    params_list = [
        gamma.fit(samples, floc=0)
        for samples in synth_mean_Ys.T
    ]
    bounds_list = [
        gamma.ppf([1 - p_1_sigma, p_1_sigma], *params)
        for params in params_list
    ]
    low, high = zip(*bounds_list)
    if plot_type == 'R':
        low *= X**3
        high *= X**3
        
    plt.fill_between(
        X, low, high, facecolor=color, alpha=fill_alpha
    )

    norm = np.array(ds).size / area / trapezoid(
        differential_pf(X), X
    )
    if plot_type == 'R':
        Y = norm * differential_pf(X) * X**3
        ylabel_type = 'R '
    elif plot_type == 'differential':
        Y = norm * differential_pf(X)
        ylabel_type='Differential '
    else:
        raise ValueError(
            'plot_type must either be \'differential\' or \'R\''
        )
    plt.plot(X, Y, color=color, lw=lw, ls=ls)
    format_cc_plot(
        X, Y, np.array(low), np.array(high),
        ylabel_type=ylabel_type, x_max_pad=0,
        fontsize=fontsize
    )

    return X, synth_mean_Ys


def calc_kde(
    ds, area, d_min=None, d_max=10**3.5, n_points=10000,
    factor=1.1
):
    if d_min is None:
        d_min = np.min(ds)
    logD = np.linspace(np.log10(d_min), np.log10(d_max), n_points)
    D = 10**logD
    ds = np.flip(np.sort(ds))
    i = 0
    log_ds = np.log10(ds[:,np.newaxis])
    kde_matrix = norm.pdf(logD, log_ds, np.log10(factor)) / area
    normalization = 1 / ds[:,np.newaxis] / math.log(10)
    kde = np.sum(kde_matrix * normalization, axis=0)
    return D, kde


def _use_version_1(synth_ds, ds, factor, n):
    synth_ds = np.sort(synth_ds)
    ds = np.sort(ds)
    nearest_ds = ds[np.abs(synth_ds[:, None] - ds).argmin(axis=1)]
    final_ds = np.ones(synth_ds.size, dtype=bool)
    for i in range(synth_ds.size):
        di, Di = synth_ds[i], nearest_ds[i]
        if Di == nearest_ds.min():
            final_ds[i] = True
        elif Di == nearest_ds.max():
            final_ds[i] = False
        else:
            left_gap = Di - nearest_ds[nearest_ds < Di].max()
            right_gap = nearest_ds[nearest_ds > Di].min() - Di
            left_buffer = n * (factor - 1 ) * (
                di + nearest_ds[nearest_ds < Di].max()
            )
            right_buffer = n * (factor - 1 ) * (
                di + nearest_ds[nearest_ds > Di].min()
            )
            if left_gap < left_buffer and right_gap < right_buffer:
                final_ds[i] = True
            else:
                final_ds[i] = False
    return final_ds


kde_bounds_dict = {}
def kde_bounds(
    ds, area, d_min=None, d_max=10**3.5, n_points=10000,
    factor=1.1, n_synths=1000, n=0.5,
    confidence=np.array([p_1_sigma, 0.977249868])
):
    args = match_args(
        locals(), kde_bounds, exclude=['ds', 'confidence']
    )
    args_key = tuple(
        list(ds) + list(confidence) + list(args.values())
    )
    if args_key in kde_bounds_dict:
        return kde_bounds_dict[args_key]
    else:
        if d_min is None:
            d_min = np.min(ds)
        ds = np.sort(ds)
        D, kde = calc_kde(**match_args(locals(), calc_kde))
        N = np.array(ds).size
        synth_d_list = synth_fixed_N(
            N=N, dmin=d_min, dmax=d_max, n_steps=n_synths,
            differential_pf=fit_pf(D, kde)
        )[0]
        crossover_d = D[_use_version_1(D, ds, factor, n)].max()
        
        def _choose_ds(synth_ds, ds, crossover_d, N):
            kde_ds = synth_ds[synth_ds >= crossover_d]
            sampled_ds = np.random.choice(
                ds[ds < crossover_d], N - kde_ds.size
            )
            return np.array(list(kde_ds) + list(sampled_ds))
        
        synth_kdes = [
            calc_kde(
                _choose_ds(synth_ds, ds, crossover_d, N), 
                area, **match_kwargs(locals(), calc_kde)
            )[1]
            for synth_ds in synth_d_list
        ]
    
        upper1, upper2 = np.percentile(
            synth_kdes, 100 * confidence, axis=0
        )
        lower1, lower2 = np.percentile(
            synth_kdes, 100 * (1 - confidence), axis=0
        )

        r_tuple = lower1, lower2, upper1, upper2
        kde_bounds_dict[args_key] = r_tuple
        return r_tuple


def plot_kde(
    ds, area, d_min=None, d_max=10**3.5, n_points=10000,
    color='black', lw=1.5, factor=1.1, plot_error=False,
    n_synths=1000, n=0.5, alpha=0.2, error_lw=0.5,
    confidence=np.array([p_1_sigma, 0.977249868]),
    xmin_factor=1.2, xmax_factor=1.5
): 
    if d_min is None:
        d_min = np.min(ds)
    D, kde = calc_kde(**match_args(locals(), calc_kde))
    plt.plot(D, kde, color=color, lw=lw)
    plt.xscale('log')
    plt.yscale('log')
    xmin = np.min(ds) / xmin_factor
    xmax = np.max(ds) * xmax_factor
    plt.xlim([xmin, xmax])
    in_range = (D >= xmin) & (D <= xmax)
    plt.ylim([kde[in_range].min(), kde[in_range].max()])
    if plot_error:
        lower1, lower2, upper1, upper2 = kde_bounds(
            **match_args(locals(), kde_bounds)
        )
        plt.fill_between(
            D, lower2, upper2, color=color, alpha=alpha, ec=None
        )
        plt.plot(D, lower1, ':', lw=error_lw, color=color)
        plt.plot(D, upper1, ':', lw=error_lw, color=color)
        plt.ylim([kde[in_range].min(), upper2[in_range].max()])


def plot_kde_R(
    ds, area, d_min=None, d_max=10**3.5, n_points=10000,
    color='black', lw=1.5, factor=1.1, plot_error=False,
    n_synths=1000, n=0.5, alpha=0.2, error_lw=0.5,
    confidence=np.array([p_1_sigma, 0.977249868]),
    xmin_factor=1.2, xmax_factor=1.5
): 
    if d_min is None:
        d_min = np.min(ds)
    D, kde = calc_kde(**match_args(locals(), calc_kde))
    kde *= D**3
    plt.plot(D, kde, color=color, lw=lw)
    plt.xscale('log')
    plt.yscale('log')
    xmin = np.min(ds) / xmin_factor
    xmax = np.max(ds) * xmax_factor
    plt.xlim([xmin, xmax])
    in_range = (D >= xmin) & (D <= xmax)
    plt.ylim([kde[in_range].min(), kde[in_range].max()])
    if plot_error:
        lower1, lower2, upper1, upper2 = kde_bounds(
            **match_args(locals(), kde_bounds)
        )
        lower1, lower2 = lower1 * D**3, lower2 * D**3
        upper1, upper2 = upper1 * D**3, upper2 * D**3
        plt.fill_between(
            D, lower2, upper2, color=color, alpha=alpha, ec=None
        )
        plt.plot(D, lower1, ':', lw=error_lw, color=color)
        plt.plot(D, upper1, ':', lw=error_lw, color=color)
        plt.ylim([kde[in_range].min(), upper2[in_range].max()])


