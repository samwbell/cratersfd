from .random_variable_module import *
from .generic_plotting_module import *


def fast_calc_cumulative_binned(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left',
    reference_point=1.0, skip_zero_crater_bins=False, d_min=None,
    start_at_reference_point=False, d_max=None, return_N=False
):

    if d_min is not None:
        reference_point = d_min
        start_at_reference_point = True
    
    bin_counts, bin_array, bin_min, bin_max = bin_craters(
        **match_args(locals(), bin_craters)
    )
    
    cumulative_count_array = np.flip(np.flip(bin_counts).cumsum())
    
    diameter_array = get_bin_parameters(
        ds, bin_counts, bin_array, x_axis_position=x_axis_position
    )
        
    if skip_zero_crater_bins:
        diameter_array = diameter_array[bin_counts != 0]
        cumulative_count_array = cumulative_count_array[bin_counts != 0]
    
    return_tuple = diameter_array, cumulative_count_array / area
    
    if return_N:
        return_tuple += (cumulative_count_array,)
    
    return return_tuple


def calc_cumulative_binned_pdfs(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left',
    skip_zero_crater_bins=False, reference_point=1.0, kind='log',
    start_at_reference_point=False, d_max=1E4, d_min=None
):
    x_array, density_array = fast_calc_cumulative_binned(
        **match_args(locals(), fast_calc_cumulative_binned)
    )
    cumulative_counts = density_array * area
    density_pdf_list = []
    for cumulative_count in cumulative_counts:
        lambda_rv = lambda_pdf(cumulative_count, kind=kind)
        density_pdf_list.append(lambda_rv / area)
    return x_array, density_pdf_list


def plot_cumulative_binned(
    ds, area, bin_width_exponent=neukum_bwe, x_axis_position='left', 
    skip_zero_crater_bins=False, reference_point=1.0, d_max=1000,
    start_at_reference_point=False, color='black', 
    alpha=1.0, plot_points=True, plot_error_bars=True,
    do_formatting=True, d_min=None, kind='log'
):
    bin_ds, pdf_list = calc_cumulative_binned_pdfs(
        **match_args(locals(), calc_cumulative_binned_pdfs)
    )
    modes = np.array([pdf.mode() for pdf in pdf_list])
    bin_ds = bin_ds[modes > 1E-100]
    pdf_list = [pdf for pdf in pdf_list if pdf.mode() > 1E-100]
    plot_pdf_list(
        bin_ds, pdf_list, 
        **match_kwargs(locals(), plot_pdf_list)
    )


