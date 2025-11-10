from .random_variable_module import *
from .pareto_module import *


def per_decade(n):
    return math.log(10, 2) / n
    
    
neukum_bwe = per_decade(18)


def safe_d_min(ds):
    second_smallest = np.sort(np.unique(ds))[1]
    return np.exp(
        np.log(np.min(ds)) - 0.5 * (
            np.log(second_smallest) - np.log(np.min(ds))
        )
    )


def bin_craters(
    ds, bin_width_exponent=neukum_bwe, reference_point=1.0, 
    start_at_reference_point=False, d_max=None, growth_rate=1.0,
    fraction=1.0, min_count=1, d_min=None
):
    if d_min is not None:
        reference_point = d_min
    if start_at_reference_point:
        bin_min = 0
    else:
        bin_min = math.ceil(
            math.log(min(ds) / reference_point, 2) / bin_width_exponent
        )
    if d_max is not None:
        if growth_rate == 1.0:
            bin_max = math.ceil(
                math.log(d_max / reference_point, 2) / bin_width_exponent
            )
        else:
            lng = math.log(growth_rate)
            term1 = math.log(d_max / reference_point, 2) / bin_width_exponent
            bin_max = math.ceil(lambertw(term1 * lng).real / lng)
    else:
        if growth_rate == 1.0:
            bin_max = math.ceil(
                math.log(max(ds) / reference_point, 2) / bin_width_exponent
            )
        else:
            lng = math.log(growth_rate)
            term1 = math.log(max(ds) / reference_point, 2) / bin_width_exponent
            bin_max = math.ceil(lambertw(term1 * lng).real / lng)
    r, w, g = reference_point, bin_width_exponent, growth_rate
    bins = [
        r * 2**(w * n * g**n) 
        for n in list(range(bin_min, bin_max + 1))
    ]
    if d_max is not None:
        bins[-1] = d_max
    if (d_min is not None) and (bins[0] > d_min):
        bins = np.insert(bins, 0, d_min)
    counts, bins = np.histogram(ds, bins)
    bins = np.append(bins[:-1][counts >= min_count], bins[-1])
    counts, bins = np.histogram(ds, bins)
    fbins = [bins[0]]
    for i in range(bins.shape[0] - 2):
        ei, ei1 = bins[i], bins[i + 1]
        lei, lei1 = np.log10(ei), np.log10(ei1)
        fbin = 10**(lei + fraction * (lei1 - lei))
        fbins.append(fbin)
    fbins.append(bins[-1])
    counts, bins = np.histogram(ds, fbins)
    if counts[0] < min_count:
        bins = np.delete(bins, 1)
        counts, bins = np.histogram(ds, bins)
    bins = np.append(bins[:-1][counts >= min_count], bins[-1])
    counts, bins = np.histogram(ds, bins)
    if (start_at_reference_point == False) and (max(ds) < bins[1]):
        raise ValueError(
            'The data cannot be binned because each crater is within '
            'the smallest bin.  Consider setting '
            'start_at_reference_point=True and setting a reference_point '
            'value for the minimum diameter counted.  Without a known '
            'minimum diameter, we must reject data in the smallest bin '
            'because the smallest bin is not fully sampled.'
        )
    return counts, bins, bin_min, bin_max


def get_bin_parameters(ds, counts, bins, x_axis_position='left'):
    
    if x_axis_position=='left':
        x_array = bins[:-1]
        
    elif x_axis_position=='log_center':
        x_array = np.sqrt(bins[:-1] * bins[1:])

    elif x_axis_position=='linear_center':
        x_array = (bins[:-1] + bins[1:]) / 2
        
    elif x_axis_position=='gmean':
        x_array = (bins[:-1] + bins[1:]) / 2
        x_array[counts != 0] = np.array([
            gmean(ds[np.digitize(ds, bins) == i]) 
            for i in np.array(range(1, len(counts) + 1))[counts != 0]
        ])
            
    return x_array


def plot_with_error(
    ds, val, lower, upper, color='black', alpha=1.0, ms=4, 
    plot_error_bars=True, plot_points=True, kind='log',  
    ylabel_type='Cumulative ', elinewidth=0.5, do_formatting=None,
    point_label=None, fontsize=14
):

    axis_exists = any(plt.gcf().get_axes())
    if not axis_exists:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()
    fig = plt.gcf()
    
    if plot_points:
        plt.plot(
            ds, val, marker='s', ls='', mfc=color, 
            mec=color, mew=1.2, ms=ms, label=point_label
        )
    
    if plot_error_bars:
        plt.errorbar(
            ds, val, yerr=[lower, upper], fmt='none', 
            color=color, alpha=alpha, elinewidth=elinewidth
        )
        
    if do_formatting is None:
        format_bool = not axis_exists
    else:
        format_bool = do_formatting
    
    if format_bool:
        format_cc_plot(
            ds, val, val - lower, val + upper,
            **match_kwargs(locals(), format_cc_plot)
        )
        

def plot_pdf_list(
    ds, pdf_list, color='black', alpha=1.0, plot_error_bars=True, 
    plot_points=True, kind='log', area=None, 
    ylabel_type='Cumulative ', ms=4, elinewidth=0.5,
    do_formatting=True, fontsize=14
):

    for i in range(len(pdf_list)):
        if kind.lower() != pdf_list[i].kind.lower():
            if kind.lower() == 'sqrt(n)':
                if area is None:
                    raise ValueError(
                        'If the kind of error bar is \'sqrt(N)\', and '
                        'the pdfs are not already sqrt(N) pdfs, the '
                        'function needs to know the area to figure '
                        'out the N to find sqrt(N).  To fix, set '
                        'area=<area value>.'
                    )
                N_pdf = pdf_list[i] * area
                pdf_list[i] = N_pdf.as_kind('sqrt(N)') / area
            else:
                pdf_list[i] = pdf_list[i].as_kind(kind)
        
    val = np.array([pdf.val for pdf in pdf_list])
    lower = np.array([pdf.lower for pdf in pdf_list])
    upper = np.array([pdf.upper for pdf in pdf_list])

    plot_with_error(**match_args(locals(), plot_with_error))


