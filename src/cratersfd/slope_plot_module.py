from .random_variable_module import *
from .generic_plotting_module import *

def calc_rolling_slopes(
    ds, n=4, end_n=0, dmin=None, dmax=None, kind='linear'
):
    if dmin is None:
        d_min = 2 * ds[0] - np.mean(ds[:2])
    else:
        d_min = dmin
    edges = (ds[1:] + ds[:-1]) / 2
    i_min = np.where(edges > d_min)[0][0]
    i_max = edges.shape[0] - (n + end_n)
    i_range = range(i_min, i_max)
    slope_rv_list = [
        slope_pdf(
            ds, dmin=edges[i - n - 1], dmax=edges[i + n]
        ).as_kind(kind)
        for i in i_range
    ]
    if dmax is None:
        end_rv = pareto_pdf(ds, dmin=edges[i_max - 1]).as_kind(kind)
    else:
        end_rv = truncated_pareto_pdf(
            ds, dmin=edges[i_max - 1], dmax=dmax
        ).as_kind(kind)
    slope_rv_list = slope_rv_list + (n + end_n + 1) * [end_rv]
    if dmax is None:
        end = 10000
    else:
        end = dmax
    edges = [d_min] + list(edges[edges > d_min]) + [end]
    return edges, slope_rv_list


def plot_slopes(
    edges, slope_rv_list, color='mediumslateblue', fill_alpha=0.3, lw=1.5
):
    vals = np.array([rv.val for rv in slope_rv_list])
    lowers = np.array([rv.lower for rv in slope_rv_list])
    uppers = np.array([rv.upper for rv in slope_rv_list])
    
    lows = vals - lowers
    highs = vals + uppers
    
    plt.hlines(vals, edges[:-1], edges[1:], color=color, lw=lw)
    plt.fill_between(
        np.repeat(edges, 2)[1 : -1], np.repeat(lows, 2), 
        np.repeat(highs, 2), facecolor=color, alpha=fill_alpha
    )
    
    plt.xscale('log')
    plt.xlim([0.01, edges[-1]])
    plt.xlabel('Crater Diameter (km)', size=12)
    plt.ylabel('Negative Slope α', size=12)


def rolling_slope_plot(
    ds, n=4, end_n=0, dmin=None, dmax=None, kind='linear', 
    color='mediumslateblue', fill_alpha=0.15, lw=1.0,
    no_printing=False
):
    if not no_printing:
        print('Calculating slope PDFs...')
    t1 = time.time()
    edges, slope_rv_list = calc_rolling_slopes(
        ds, n=n, end_n=end_n, dmin=dmin, dmax=dmax, kind=kind
    )
    plot_slopes(
        edges, slope_rv_list, color=color, 
        fill_alpha=fill_alpha, lw=lw
    )
    t2 = time.time()
    if not no_printing:
        print('Done in ' + format_runtime(t2 - t1))

    return edges, slope_rv_list


def split_into_bins(ds, bin_size=25):
    n_ds = ds.shape[0]
    n_bins = n_ds // bin_size  
    remainder = n_ds % bin_size

    if remainder == 0:
        bins = [ds[i : i + bin_size] for i in range(0, n_ds, bin_size)]
    else:
        bin0_size = bin_size + remainder
        bins = [ds[:bin0_size]] + [
            ds[i : i + bin_size] for i in range(bin0_size, n_ds, bin_size)
        ]

    return bins


def calc_binned_slopes(
    raw_ds, bin_size=25, bin_edges=None, dmin=None, dmax=None,
    rollover=None, kind='linear'
):
    ds = np.sort(raw_ds)
    if dmin is None:
        d_min = 2 * ds[0] - np.mean(ds[:2])
    else:
        d_min = dmin
    if dmax is None:
        if bin_edges is None:
            d_max = 10000
        else:
            d_max = bin_edges[-1]
    else:
        d_max = dmax
    
    if bin_edges is None:
        if rollover is None:
            rollover = np.mean(ds[-1 * bin_size - 1 : -1 * bin_size + 1])
        bins = split_into_bins(ds[ds <= rollover], bin_size=bin_size)
        edges = np.array([d_min] + [
            np.mean([bins[i][-1], bins[i + 1][0]]) 
            for i in range(len(bins) - 1)
        ] + [rollover, d_max])
    else:
        edges = bin_edges

    if dmax is None:
        end_rv = pareto_pdf(ds, dmin=edges[-2]).as_kind(kind)
    else:
        end_rv = truncated_pareto_pdf(
            ds, dmin=edges[-2], dmax=edges[-1]
        ).as_kind(kind)

    slope_rv_list = [
        slope_pdf(ds, dmin=edges[i], dmax=edges[i + 1]).as_kind(kind) 
        for i in range(len(edges) - 2)
    ] + [end_rv]
    
    return edges, slope_rv_list


def binned_slope_plot(
    ds, bin_size=25, bin_edges=None, dmin=None, dmax=None, rollover=None,
    color='mediumslateblue', fill_alpha=0.15, lw=1.0, no_printing=False,
    kind='linear'
):
    if not no_printing:
        print('Calculating slope PDFs...')
    t1 = time.time()
    edges, slope_rv_list = calc_binned_slopes(
        ds, bin_size=bin_size, bin_edges=bin_edges, dmin=dmin, 
        dmax=dmax, rollover=rollover, kind=kind
    )
    plot_slopes(
        edges, slope_rv_list, color=color, 
        fill_alpha=fill_alpha, lw=lw
    )
    t2 = time.time()
    if not no_printing:
        print('Done in ' + format_runtime(t2 - t1))
    return edges, slope_rv_list


