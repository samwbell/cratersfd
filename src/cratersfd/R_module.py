from .differential_module import *
from .random_variable_module import *
from .generic_plotting_module import *

def fast_calc_R(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None,
    start_at_reference_point=False, growth_rate=1.0,
    fraction=1.0, pf=npf_new_loglog, do_correction=True,
    full_output=False, x_axis_position='log_center'
):
    args = match_args(locals(), fast_calc_dif)
    args['full_output'] = True
    sorted_ds, difs, counts, widths, bins, cfs = fast_calc_dif(**args)
    rfs = (bins[:-1] * bins[1:])**1.5
    Rs = rfs * difs

    if full_output:
        return sorted_ds, Rs, counts, widths, bins, cfs, rfs
    else:
        return sorted_ds, Rs


def calc_R_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None,
    start_at_reference_point=False, growth_rate=1.0,
    fraction=1.0, pf=npf_new_loglog, do_correction=True,
    x_axis_position='log_center', kind='log'
):
    args = match_args(locals(), fast_calc_R)
    args['full_output'] = True
    sorted_ds, Rs, counts, widths, bins, cfs, rfs = fast_calc_R(**args)
    lambda_pdfs = lambda_pdf(counts, kind=kind)
    R_rv_list = rfs * cfs / widths / area * lambda_pdfs
    return sorted_ds, R_rv_list


def plot_R(
    ds, area, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    min_count=1, d_max=None, d_min=None,
    start_at_reference_point=False, growth_rate=1.0,
    fraction=1.0, pf=npf_new_loglog, 
    do_correction=True, ax='None', color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True,
    x_axis_position='log_center', ms=4, kind='log'
):
    args = match_args(locals(), calc_R_pdfs)
    bin_ds, pdf_list = calc_R_pdfs(**args)
    args = match_args(locals(), plot_pdf_list)
    args['ds'] = bin_ds
    plot_pdf_list(**args, ylabel_type='R')


def calc_sash_R(
    ds, area, d_min=None, 
    bin_width_exponent=neukum_bwe, d_max=None, 
    growth_rate=1.3, n_points=10000, n_shifts=200,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    return_lines=False
):
    d, dif, difs = calc_sash(**match_args(locals(), calc_sash))[:3]
    if return_lines:
        return d, dif * d**3, difs * d**3
    else:
        return d, dif * d**3


def plot_sash_R(
    ds, area, d_min=None, 
    bin_width_exponent=per_decade(18), d_max=None, 
    growth_rate=1.2, n_points=10000, n_shifts=200,
    color='mediumslateblue', plot_lines=False, lw=1.2,
    line_color='mediumslateblue', line_lw=0.2,
    min_count=1, n_iterations=5, n_alpha_points=10000,
    plot_error=True, fill_alpha=0.15, kernel=None, 
    reduction_factor=1.0, error_downsample=10, kind='log',
    error_bin_width_exponent=per_decade(18)
):
    t1 = time.time()
    args = match_args(locals(), calc_sash_R)
    if plot_lines:
        X, mean_Y, Ys = calc_sash_R(**args, return_lines=True)
    else:
        X, mean_Y = calc_sash_R(**args)
    t2 = time.time()
    print('Calculation time: ' + format_runtime(t2 - t1))
    
    if plot_lines:
        for Y in Ys:
            plt.plot(X, Y, color=line_color, lw=line_lw)

    plt.plot(X, mean_Y, color, lw=lw)

    Ycut = mean_Y[X <= 100 * np.max(ds)]
    Xcut = X[X <= 100 * np.max(ds)]

    if plot_error:
        Xp = Xcut[::error_downsample]
        Yp = Ycut[::error_downsample]
        C = cumulative_trapezoid(Yp / Xp**3, Xp, initial=0) * area
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
            Xp, val, low, high, ylabel_type='R ',
            x_max_pad=0
        )

    else:

        format_cc_plot(
            Xcut, Ycut, Ycut, Ycut, ylabel_type='R ',
            x_max_pad=0
        )
    
    return X, mean_Y


