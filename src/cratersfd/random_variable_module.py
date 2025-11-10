from .random_variable_backend_module import *


def round_dec(dec, dn, rounding_n):
    if np.isnan(dec):
        dec_str = 'None'
    else:
        dec_str = str(round(float(dec / 10**dn), rounding_n))
        pre_dot_len = len(dec_str.split('.')[0]) + 1
        dec_str += '0' * (rounding_n + pre_dot_len - len(dec_str))
    return dec_str
    

def plot_label(
    rounding_n, low, high, val, X, P, xlim,
    label_shift_x, label_shift_y, upshift,
    force_label_side, label_text_size, pdf_label,
    color, label_color, unit, pdf_label_size,
    pdf_gap_shift, dn, label_x, label_y, mf,
    before_label
):
    ax = plt.gca()
    fig = plt.gcf()
    v0 = 0
    if dn is None:
        if val == 0:
            dn = 0
            v0 = 1
        elif np.isnan(val):
            dn = 0
        else:
            dn = int(np.floor(np.log10(np.abs(val))))
    if mf and (val != 0):
        if dn == 0:
            val_str = round_dec(val, 0, rounding_n + v0)
            upper_str = round_dec(high / val, 0, rounding_n + v0)
            lower_str = round_dec(val / low, 0, rounding_n + v0)
            exp_str = ''
        elif dn == -1:
            val_str = round_dec(val, 0, rounding_n + 1)
            upper_str = round_dec(high / val, 0, rounding_n + 1)
            lower_str = round_dec(val / low, 0, rounding_n + 1)
            exp_str = ''
        elif dn == 1:
            val_str = round_dec(val, 0, rounding_n - 1)
            upper_str = round_dec(high / val, 0, rounding_n - 1)
            lower_str = round_dec(val / low, 0, rounding_n - 1)
            exp_str = ''
        else:
            val_str = round_dec(val, dn, rounding_n)
            upper_str = round_dec(high / val, 0, rounding_n)
            lower_str = round_dec(val / low, 0, rounding_n)
            exp_str = rf'×10$^{{{dn}}}$'
        upper_str = '×' + upper_str
        lower_str = '÷' + lower_str
    else:
        if dn == 0:
            val_str = round_dec(val, 0, rounding_n + v0)
            upper_str = round_dec(high - val, 0, rounding_n + v0)
            lower_str = round_dec(val - low, 0, rounding_n + v0)
            exp_str = ''
        elif dn == -1:
            val_str = round_dec(val, 0, rounding_n + 1)
            upper_str = round_dec(high - val, 0, rounding_n + 1)
            lower_str = round_dec(val - low, 0, rounding_n + 1)
            exp_str = ''
        elif dn == 1:
            val_str = round_dec(val, 0, rounding_n - 1)
            upper_str = round_dec(high - val, 0, rounding_n - 1)
            lower_str = round_dec(val - low, 0, rounding_n - 1)
            exp_str = ''
        else:
            val_str = round_dec(val, dn, rounding_n)
            upper_str = round_dec(high - val, dn, rounding_n)
            lower_str = round_dec(val - low, dn, rounding_n)
            exp_str = rf'×10$^{{{dn}}}$'
        upper_str = '+' + upper_str
        lower_str = '-' + lower_str
    num_str = rf'${val_str}_{{{lower_str}}}^{{{upper_str}}}$'
    label_str = before_label + num_str + exp_str
    if unit is not None:
        label_str += unit
    
    min_X = xlim[0]
    max_X = xlim[1]
    peak_X = X[np.argmax(P)]
    if force_label_side is None:
        if (peak_X - min_X) < (max_X - peak_X):
            label_side = 'right'
            pdf_label_side = 'left'
        else:
            label_side = 'left'
            pdf_label_side = 'right'
    else:
        label_side = force_label_side
    if ax.spines[label_side].get_visible():
        buffer = 0.007
    else:
        buffer = 0
    text_x_dict = {
        'left' : min_X + buffer * (max_X - min_X),
        'right' : max_X - buffer * (max_X - min_X)
    }
    if label_x is None:
        text_x = text_x_dict[label_side] + label_shift_x
    else:
        text_x = label_x
    if label_color=='same':
        l_color = color
    else:
        l_color = label_color
    label_text = plt.text(
        text_x, upshift, label_str, ha=label_side, va='bottom',
        size=label_text_size, color=l_color
    )
    if label_side == 'right':
        x0 = min_X + 0.8 * (max_X - min_X)
    else:
        x0 = min_X + 0.2 * (max_X - min_X)
    y0 = np.interp(x0, X, P)
    if label_y is None:
        text_y = y0 + 0.03 * (P.max() - upshift) + label_shift_y
    else:
        text_y = label_y
    
    if pdf_label is not None:
        if pdf_label_size is None:
            _pdf_label_size = label_text_size - 1
        else:
            _pdf_label_size = pdf_label_size
        pdf_text = plt.text(
            text_x, text_y, pdf_label, ha=label_side, va='bottom',
            size=_pdf_label_size, color=l_color
        )
        text_y = y0 + 0.25 * (P.max() - upshift) + label_shift_y
        text_y += pdf_gap_shift
        
    label_text.set_position((text_x, text_y))
    fig.canvas.draw()
    
    
def fix_start(X, P, fixed_start_x, fixed_start_p):
    P = P[X > fixed_start_x]
    X = X[X > fixed_start_x]
    min_X = fixed_start_x
    X = np.insert(X, 0, fixed_start_x)
    if fixed_start_p is not None:
        P = np.insert(P, 0, fixed_start_p)
    else:
        P = np.insert(P, 0, round(P[0]))
    return X, P, min_X
    

def erase_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.yticks([])


def format_cc_plot(
    d_array, full_density, full_low=None, full_high=None, 
    full_ds=None, ylabel_type='Cumulative ', kind='log',
    x_max_pad=0.1, fontsize=14
):
    
    plt.xscale('log')
    plt.yscale('log')

    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)

    if full_low is None:
        full_low = full_density
    if full_high is None:
        full_high = full_density

    xmax = np.max(d_array)
    if full_ds is not None:
        xmin = np.min(full_ds)
    else:
        xmin = np.min(d_array) 
    xrange = np.log10(xmax / xmin)
    plt.xlim([
        xmin / (10**(0.05 * xrange)), xmax * 10**(x_max_pad * xrange)
    ])

    ymax = np.nanmax(full_high)
    if kind.lower() == 'sqrt(n)':
        ymin = np.nanmin(full_density) / 10
    else:
        ymin = np.nanmin(full_low[full_low > 0])
    yrange = np.log10(ymax / ymin)
    plt.ylim([ymin / (10**(0.05 * yrange)), ymax * 10**(0.05 * yrange)])

    plt.ylabel(ylabel_type + rf' Crater Density (km$^{{-2}}$)', size=fontsize)
    plt.xlabel('Crater Diameter (km)', size=fontsize)

    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.25, color='gray')

    
class RandomVariable(MathRandomVariable):
    
    
    def plot(
        self, upshift=0, color='mediumslateblue', 
        fixed_start_x=None, fixed_start_p=None, label=False, 
        rounding_n=2, label_shift_x=0, label_shift_y=0, unit=None, 
        label_text_size=10, force_label_side=None, xlim=None, 
        kind='same', label_color='same', alpha=0.3,
        pdf_label=None, standardized=True, force_erase_box=None,
        pdf_label_size=None, pdf_gap_shift=0, dn=None,
        label_x=None, label_y=None, lw=2, mf=None, 
        plot_error_bars=True, edge_lines=False,
        before_label='', **kwargs
    ):
  
        axis_exists = any(plt.gcf().get_axes())
        if not axis_exists:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(111)
            if force_erase_box is None:
                erase_box(ax)
        else:
            ax = plt.gca()
        if force_erase_box:
            erase_box(ax)
        fig = plt.gcf()
        
        X, P, C = self.X, self.P, self.C()
        if standardized:
            P = P / P.max()
        if fixed_start_x is not None:
            X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        P = P + upshift
        if X[0] > X[-1]:
            X = np.flip(X)
            P = np.flip(P)
        
        plt.plot(X, P, color, linewidth=lw)
        if xlim is not None:
            plt.xlim(xlim)
        xlim = ax.get_xlim()

        if kind.lower() not in {'same', self.kind.lower()}:
            krv = self.as_kind(kind)
            low, val, high = krv.low, krv.val, krv.high
        else:
            low, val, high = self.low, self.val, self.high
            kind = self.kind
        
        if plot_error_bars:
            if np.isnan(low):
                ilow = np.min(X)
            else:
                ilow = low
            if np.isnan(high):
                ihigh = np.max(X)
            else:
                ihigh = high
            interp_n = np.max([np.sum((X > ilow) & (X < ihigh)), 13000])
            X_interp = np.linspace(ilow, ihigh, interp_n)
            P_interp = np.interp(X_interp, X, P)
            plt.fill_between(
                X_interp, upshift, P_interp, 
                facecolor=color, alpha=alpha
            )
            low_P = np.interp(low, X_interp, P_interp)
            high_P = np.interp(high, X_interp, P_interp)
            val_P = np.interp(val, X_interp, P_interp)
            plt.plot([val, val], [upshift, val_P], color=color)
            if edge_lines:
                plt.plot([low, low], [upshift, low_P], ':', color=color)
                plt.plot([high, high], [upshift, high_P], ':', color=color)

        if mf is None:
            if kind.lower() in {'log', 'auto log'}:
                _mf = True
            else:
                _mf = False
        else:
            _mf = mf

        if label:
            plot_label(
                rounding_n, low, high, val, X, P, xlim,
                label_shift_x, label_shift_y, upshift,
                force_label_side, label_text_size, pdf_label,
                color, label_color, unit, pdf_label_size,
                pdf_gap_shift, dn, label_x, label_y, _mf,
                before_label
            )

    def plot_differential(self, **kwargs):
        plt.plot(self.X, self.P, **match_kwargs(locals(), plt.plot))
        format_cc_plot(
            self.X, self.P,
            ylabel_type='Differential ', x_max_pad=0,
            **match_kwargs(locals(), format_cc_plot)
        )

    
    
def factor_pdf(
    factor, n_stds=6, n_points=10000, kind='mean',
    spacing='linear'
):
    s = np.log(factor)
    scale = np.exp(-0.5 * s**2) 
    Xmin = scale * factor**(-1 * n_stds)
    Xmax = scale * factor**n_stds
    if spacing == 'linear':
        X = np.linspace(Xmin, Xmax, n_points)
    elif spacing == 'log':
        X = np.logspace(
            np.log10(Xmin), np.log10(Xmax), n_points
        )
    else:
        raise ValueError(
            'The spacing must be either \'linear\''
            'or \'log\'.'
        )
    P = lognorm.pdf(X, s=s, scale=scale)
    return RandomVariable(X, P, kind=kind)



def apply_factor(
    n, factor, n_stds=6, n_points=None, kind='mean'
):
    if factor == 1.0:
        return n
    else:
        if n_points is None:
            if isinstance(n, MathRandomVariable):
                n_points = n.X.shape[0]
            else:
                n_points = 10000
        factor_rv = factor_pdf(
            **match_args(locals(), factor_pdf)
        )
        return n * factor_rv



def factor_func(n, factor, n_stds=6, n_points=None):
    s = np.log(factor)
    scale = n * np.exp(-0.5 * s**2)
    def func(x):
        return lognorm.pdf(x, s=s, scale=scale)
    return func



def true_d_func(d, factor):
    s = np.log(factor)
    def func(x):
        scale = np.exp(-0.5 * s**2 + np.log(x))
        return lognorm.pdf(d, s=s, scale=scale)
    return func



true_d_rv_dict = {}
def true_d_pdf(
    d, factor, n_stds=6, n_points=10000, kind='mean'
):
    args = match_args(locals(), true_d_pdf)
    key = tuple(args.values())
    if key in true_d_rv_dict:
        true_d_rv = true_d_rv_dict[key]
    else:
        Xmin = d * factor**(-1 * n_stds)
        Xmax = d * factor**n_stds
        X = np.linspace(Xmin, Xmax, n_points)
        s = np.log(factor)
        scale = X * np.exp(-0.5 * s**2)
        likelihood = lognorm.pdf(d, s=s, scale=scale)
        w = 1 / X**2
        P = w * likelihood
        true_d_rv = RandomVariable(X, P, kind=kind)
        true_d_rv_dict[key] = true_d_rv
    return true_d_rv



def sample_factor_pdf(n, factor, size):
    sigma = np.log(factor)
    mean = -0.5 * sigma**2 + np.log(n)
    if factor == 1:
        samples = np.ones(size)
    else:
        samples = np.random.lognormal(
            mean=mean, sigma=sigma, size=size
        )
    return samples
    
    
    
def add_lognormal_ps(p1, p2=None):
    if type(p1) in {list, np.ndarray}:
        ps = np.array(p1)
        return np.exp(np.sqrt(np.sum(np.log(1 + ps)**2)))
    elif p2 is None:
        raise ValueError(
            'If the first input is not a list or array, you must '
            'input a second percentage value to add to the first.'
        )
    else:
        sum_of_squares = np.log(1 + p1)**2 + np.log(1 + p2)**2
        return np.exp(np.sqrt(sum_of_squares)) - 1



def subtract_lognormal_ps(p1, p2):
    return np.exp(np.sqrt(np.log(1 + p1)**2 - np.log(1 + p2)**2)) - 1
    
    

def lambda_error_lognormal(
    N, random=1.5, systematic=1.1, additional=1.1
):
    if N < 1:
        N = 1
    return np.exp(np.sqrt(np.sum([
        np.log((np.exp(np.log(random)**2) - 1) / N + 1),
        np.log(systematic)**2, np.log(additional)**2
    ])))



lambda_with_error_dict = {}
def lambda_with_error(
    X, N, random=1.5, systematic=1.1, additional=1.1
):
    args = match_args(locals(), lambda_error_lognormal)
    key = tuple(args.values())
    if key in lambda_with_error_dict:
        lambda_with_error = lambda_with_error_dict[key].match_X(X)
    else:
        lambda_rv = RandomVariable(X, gamma.pdf(X, N + 1))
        factor = lambda_error_lognormal(**args)
        lambda_with_error = lambda_rv * apply_factor(1, factor)
        lambda_with_error = lambda_with_error.match_X(X)
        lambda_with_error_dict[key] = lambda_with_error
    return lambda_with_error



def lambda_pdf_from_N_pmf(
    N_array, pmf, cum_prob_edge=1E-7, n_points=10000,
    random=1.5, systematic=1.1, additional=1.1,
    p_cutoff=1E-10, apply_error=False
):

    N_array = N_array[pmf > p_cutoff * np.max(pmf)]
    pmf = pmf[pmf > p_cutoff * np.max(pmf)]
    
    N_min = N_array.min()
    N_max = N_array.max()

    error_exists = (
        random != 1.0 or systematic != 1.0 or additional != 1.0
    )
    X_min = gamma.ppf(cum_prob_edge, N_min + 1)
    X_max = gamma.ppf(1 - cum_prob_edge, N_max + 1)
    if apply_error and error_exists:
        kwargs = match_kwargs(locals(), lambda_error_lognormal)
        max_factor = lambda_error_lognormal(N_max, **kwargs)
        max_rv = apply_factor(1, max_factor, n_points=100)
        X_max *= max_rv.percentile(0.99)
        min_factor = lambda_error_lognormal(N_min, **kwargs)
        min_rv = apply_factor(1, min_factor, n_points=100)
        X_min *= min_rv.percentile(0.01)

    if (
        gamma.pdf(X_min, N_min + 1) > 0.001 
        and gamma.pdf(1E-150, N_min + 1) < 0.001
    ):
        X_min_search = np.logspace(-150, np.log10(X_min), 1000)
        X_min = np.interp(
            0.001, gamma.pdf(X_min_search, N_min + 1), X_min_search
        )

    X_min = max(X_min, X_max / n_points)
    X = np.linspace(X_min, X_max, n_points, endpoint=True)

    if apply_error and error_exists:
        lambda_matrix = np.array([
            lambda_with_error(X, N, **kwargs).match_X(X).P * weight 
            for N, weight in zip(N_array, pmf)
        ])
    else:
        lambda_matrix = np.array([
            gamma.pdf(X, N + 1) * weight 
            for N, weight in zip(N_array, pmf)
        ])
        
    return RandomVariable(X, lambda_matrix.sum(axis=0)).pad_with_0s()
    
    
    
class DiscreteRandomVariable:
    
    def __init__(self, X, P, val=None, low=None, high=None, kind='mean'):
        
        self.X = np.array(X)
        self.P = np.array(P)
        self.kind = kind
        self.val = val
        self.low = low
        self.high = high

        if None in {self.val, self.low}:
            self.lower = None
        else:
            self.lower = self.val - self.low

        if None in {self.val, self.high}:
            self.upper = None
        else:
            self.upper = self.high - self.val

        if kind is None:
            self.val = None
            self.low = None
            self.high = None
            self.lower = None
            self.upper = None

        elif type(kind) != str:
            raise ValueError(
                'kind must be a string: \'log\', \'auto log\', '
                '\'linear\', \'median\', \'mean\' or \'sqrt(N)\''
            )
            
        elif kind.lower() == 'log':
            if self.val in {None, np.nan}:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan} or high in {None, np.nan}:
                log_lower, log_upper = error_bar_log(
                    self.X, self.P, max_likelihood=self.val
                )
            if low in {None, np.nan}:
                self.low = 10**(np.log10(self.val) - log_lower)
            if high in {None, np.nan}:
                self.high = 10**(np.log10(self.val) + log_upper)

        elif kind.lower() in {'auto log', 'auto_log'}:
            if {None, np.nan} & {low, val, high}:
                log_max, log_lower, log_upper = fit_log_of_normal(
                    self.X, self.P
                )
            if self.val in {None, np.nan}:
                self.val = 10**log_max
            if low in {None, np.nan}:
                self.low = 10**(log_max - log_lower)
            if high in {None, np.nan}:
                self.high = 10**(log_max + log_upper)

        elif kind.lower() == 'linear':
            if self.val is None:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan} or high in {None, np.nan}:
                lower, upper = error_bar_linear(
                    self.X, self.P, max_likelihood=self.val
                )
            if low in {None, np.nan}:
                self.low = self.val - lower
            if high in {None, np.nan}:
                self.high = self.val + upper

        elif kind.lower() in {'median', 'percentile'}:
            if {None, np.nan} & {low, val, high}:
                _low, _val, _high = self.percentile([
                    1 - p_1_sigma, 0.5, p_1_sigma
                ])
            if self.val in {None, np.nan}:
                self.val = _val
            if self.low in {None, np.nan}:
                self.low = _low
            if self.high in {None, np.nan}:
                self.high = _high

        elif kind.lower() == 'mean':
            if {None, np.nan} & {low, val, high}:
                _low, _high = self.percentile([
                    1 - p_1_sigma, p_1_sigma
                ])
            if self.val in {None, np.nan}:
                self.val = self.mean()
            if self.low in {None, np.nan}:
                self.low = _low
            if self.high in {None, np.nan}:
                self.high = _high

        elif kind.lower() == 'moments':
            if self.val is None:
                self.val = rv_mean_XP(self.X, self.P)
            if low in {None, np.nan} or high in {None, np.nan}:
                self.std = rv_std_XP(self.X, self.P)
                self.skewness = rv_skewness_XP(self.X, self.P)
            if low in {None, np.nan}:
                self.low = self.val - self.std
            if high in {None, np.nan}:
                self.high = self.val + self.std

        elif kind.lower() in {'sqrt(n)', 'sqrtn', 'sqrt n'}:
            if self.val is None:
                self.val = self.X[np.argmax(self.P)]
            if low in {None, np.nan}:
                self.low = self.val - np.sqrt(self.val)
            if high in {None, np.nan}:
                self.high = self.val + np.sqrt(self.val)

        else:
            raise ValueError(
                'kind must be: \'log\', \'auto log\', '
                '\'linear\', \'median\', \'mean\', '
                '\'moments\', \'sqrt(N)\', or None'
            )

        self.lower = self.val - self.low
        self.upper = self.high - self.val

    def C(self):
        return np.cumsum(self.P) / np.sum(self.P)

    def percentile(self, p):
        v = np.interp(p, self.C(), self.X)
        _p = np.array(p)
        is_scalar = v.shape == ()
        if is_scalar:
            v = v.reshape((1,))
        v[(_p < 0) | (_p > 1)] = np.nan
        if is_scalar:
            v = v[0]
        return v
    
    def normalize(self):
        return self.__class__(self.X, self.P / self.P.sum())

    def standardize(self):
        return self.__class__(self.X, self.P / self.P.max())
    
    def lambda_pdf(
        self, cum_prob_edge=1E-7, n_points=10000,
        random=1.5, systematic=1.1, additional=1.1,
        apply_error=False
    ):
        return lambda_pdf_from_N_pmf(
            self.X, self.P, 
            **match_kwargs(locals(), lambda_pdf_from_N_pmf)
        )
    
    def mean(self):
        return np.average(self.X, weights=self.P)
    
    def std(self):
        return np.sqrt(np.cov(self.X, aweights=self.P, bias=True))

    def plot(
        self, standardized=True, no_box=True,
        upshift=0, color='mediumslateblue', 
        fixed_start_x=None, fixed_start_p=None, show_label=False, 
        rounding_n=2, label_shift_x=0, label_shift_y=0, unit=None, 
        label_text_size=10, force_label_side=None, xlim=None, 
        kind='same', label_color='same', alpha=0.3, pdf_label=None,
        pdf_label_size=None, pdf_gap_shift=0, dn=None,
        label_x=None, label_y=None, mf=None, **kwargs
    ):
        
        axis_exists = any(plt.gcf().get_axes())
        if not axis_exists:
            fig = plt.figure(figsize=(5, 2))
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        
        X, P = self.X, self.P
        if standardized:
            P = P / P.max()
        if fixed_start_x is not None:
            X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        P = P + upshift
        if X[0] > X[-1]:
            X = np.flip(X)
            P = np.flip(P)

        plt.plot(X, P, '.', color=color, **kwargs)
        if no_box:
            erase_box(plt.gca())
        
        if xlim is not None:
            plt.xlim(xlim)
        xlim = ax.get_xlim()

        if kind.lower() not in {'same', self.kind.lower()}:
            krv = self.as_kind(kind)
            low, val, high = krv.low, krv.val, krv.high
        else:
            low, val, high = self.low, self.val, self.high
            kind = self.kind

        if mf is None:
            if kind.lower() in {'log', 'auto log'}:
                mf = True
            else:
                mf = False

        if show_label:
            plot_label(**match_args(locals(), plot_label))

        

true_error_dict = {}
def true_error_pdf_single(
    N, n_points=10000, cum_prob_edge=1E-7, kind='log',
    random=1.5, systematic=1.1, additional=1.1, apply_error=False
):
    
    if tuple([N, n_points, cum_prob_edge, kind]) in true_error_dict:
        
        return_rv = true_error_dict[tuple([
            N, n_points, cum_prob_edge, kind, random, systematic,
            additional, apply_error
        ])]

    elif isinstance(N, DiscreteRandomVariable):

        return_rv = N.lambda_pdf(
            **match_args(locals(), lambda_pdf_from_N_pmf)
        )
    
    else:
    
        X, P = true_error_pdf_XP(
            N, n_points=n_points, cum_prob_edge=cum_prob_edge
        )
        
        val, lower, upper = get_error_bars(
            N, log_space=False, kind=kind, return_val=True
        )
        low = val - lower
        high = val + upper

        return_rv = RandomVariable(
            X, P, val=val, low=low, high=high, kind=kind
        )

        errors_exist = (
            random != 1.0 
            and systematic != 1.0
            and additional != 1.0
        )
        if apply_error and errors_exist:
            return_rv *= apply_factor(1, lambda_error_lognormal(
                **match_args(locals(), lambda_error_lognormal)
            ))

        true_error_dict[tuple([
            N, n_points, cum_prob_edge, kind, random, systematic,
            additional, apply_error
        ])] = return_rv
    
    return return_rv


def true_error_pdf(
    N, n_points=10000, cum_prob_edge=1E-7, kind='log',
    random=1.5, systematic=1.1, additional=1.1, apply_error=False
):
    if type(N) in {np.ndarray, list, set}:
        return np.array([true_error_pdf_single(
            Ni, **match_kwargs(locals(), true_error_pdf_single)
        ) for Ni in N])
    else:
        return true_error_pdf_single(
            **match_args(locals(), true_error_pdf_single)
        )

lambda_pdf = true_error_pdf


def sqrt_N_error_pdf(N):
    sqrtn_lambda = np.linspace(-3, 5 + N + 5 * np.sqrt(N), 10000)
    sqrtn_P = norm.pdf(sqrtn_lambda, loc=N, scale=np.sqrt(N))
    low = N - np.sqrt(N)
    high = N + np.sqrt(N)
    return RandomVariable(
        sqrtn_lambda, sqrtn_P, val=N, low=low, high=high, 
        kind='sqrt(N)'
    )


def get_bin_info(sample_array, min_val, max_val, n_bins, n_bins_baseline,
                 slope_data):
    
    sample_min, sample_max = np.min(sample_array), np.max(sample_array)
    
    if min_val == 'Auto':
        min_X = sample_min - 0.1 * (sample_max - sample_min)
        if np.min(sample_array) > 0 and min_X < 0:
            min_X = 0
    else:
        min_X = min_val
        
    if max_val == 'Auto':
        if slope_data:
            max_X = np.max(sample_array[sample_array < 0])
        else:
            max_X = sample_max + 0.1 * (sample_max - sample_min)
    else:
        max_X = max_val
        
    if n_bins == 'Auto':
        X_P_min = np.percentile(sample_array, 1)
        X_P_max = np.percentile(sample_array, 99)
        range_ratio = (max_X - min_X) / (X_P_max - X_P_min)
        n_bins_used = int(round(n_bins_baseline * range_ratio))
    else:
        n_bins_used = n_bins
        
    bins = np.linspace(min_X, max_X, n_bins_used)
    
    return bins, min_X, max_X


def make_pdf_from_samples(
    samples, n_bins='Auto', n_points=100000, min_val='Auto', max_val='Auto',
    fixed_start_x=None, fixed_start_p=None, n_bins_baseline=50, 
    slope_data=False, drop_na=True, kind='auto log'
):
    
    sample_array = np.array(samples)
    if drop_na:
        sample_array = sample_array[~np.isnan(sample_array)]
    
    bins, min_X, max_X = get_bin_info(
        sample_array, min_val, max_val, n_bins, n_bins_baseline, slope_data
    )
    
    P, bin_edges = np.histogram(sample_array, bins=bins, density=True)
    P = P / max(P)
    bin_edges = np.array(bin_edges)
    X = (bin_edges[1:] + bin_edges[:-1]) / 2
    if fixed_start_x is not None:
        X, P, min_X = fix_start(X, P, fixed_start_x, fixed_start_p)
        
    X_interp = np.linspace(min_X, max_X, n_points)
    P_interp = np.interp(X_interp, X, P)

    if slope_data and kind=='auto log':
        log_max, log_low, log_high = fit_slope_pdf(X, P)
        high = -1 * 10**(log_max - log_low)
        low = -1 * 10**(log_max + log_high)
        val = -1 * 10**log_max
        return_rv = RandomVariable(
            X_interp, P_interp, low=low, high=high, val=val, kind=kind
        )
    else:
        return_rv = RandomVariable(X_interp, P_interp, kind=kind)
        if kind.lower() == 'median':
            ps = [100 - 100 * p_1_sigma, 50.0, 100 * p_1_sigma]
            low, val, high = np.percentile(sample_array, ps)
            return_rv = RandomVariable(
                X_interp, P_interp, val=val, low=low, high=high,
                kind=kind
            )
        else:
            return_rv = RandomVariable(X_interp, P_interp, kind=kind)

    return return_rv


def ash_pdf(data, nbins=25, nshifts=10, kind='mean'):
    bins, heights = ash.ash1d(data, nbins, nshifts)
    if kind == 'mean':
        val = np.mean(data)
        low = np.percentile(data, 100 * (1 - p_1_sigma))
        high = np.percentile(data, 100 * p_1_sigma)
    else:
        val, low, high = None, None, None
    return RandomVariable(
        bins, heights, kind=kind, val=val, low=low, high=high
    )


def logspace_normal_pdf(log_max, log_lower, log_upper,
                        negative=False):
    X_min = log_max - 5 * log_lower
    X_max = log_max + 5 * log_upper
    X = np.linspace(X_min, X_max, 10000)
    X_left = X[X < log_max]
    X_right = X[X >= log_max]
    left = norm.pdf(X_left, log_max, log_lower) / norm.pdf(
            X, log_max, log_lower).max()
    right = norm.pdf(X_right, log_max, log_upper) / norm.pdf(
            X, log_max, log_upper).max()
    P = np.append(left, right)
    if negative:
        rv = RandomVariable(
            -1 * 10**X, P, val=-1 * 10**log_max, 
            high=-1 * 10**(log_max - log_lower),
            low=-1 * 10**(log_max + log_upper)
        )
    else:
        rv = RandomVariable(
            10**X, P, val=10**log_max, 
            low=10**(log_max - log_lower),
            high=10**(log_max + log_upper)
        )
    return rv



def _get(param, i):
    if type(param) in {list, np.array, set}:
        return param[i]
    else:
        return param

def plot_pdfs(
    rvs, color=cs, fixed_start_x=None, fixed_start_p=None, label=False, 
    rounding_n=2, label_shift_x=0, label_shift_y=0, unit=None, 
    label_text_size=10, force_label_side=None, xlim=None, 
    kind='same', label_color='same', alpha=0.07,
    pdf_label=None, standardize=True, force_erase_box=None,
    pdf_label_size=None, pdf_gap_shift=0, dn=None
 ):
    for i in range(len(rvs)):
        rvs[i].plot(
            upshift = 1.1 * (len(rvs) - i - 1), xlim=xlim,
            color=_get(color, i),
            fixed_start_x=_get(fixed_start_x, i), 
            fixed_start_p=_get(fixed_start_p, i), 
            label=_get(label, i), 
            rounding_n=_get(rounding_n, i), 
            label_shift_x=_get(label_shift_x, i), 
            label_shift_y=_get(label_shift_y, i), 
            unit=_get(unit, i), 
            label_text_size=_get(label_text_size, i), 
            force_label_side=_get(force_label_side, i), 
            kind=_get(kind, i), 
            label_color=_get(label_color, i), 
            alpha=_get(alpha, i),
            pdf_label=_get(pdf_label, i), 
            standardize=_get(standardize, i), 
            force_erase_box=_get(force_erase_box, i), 
            pdf_label_size=_get(pdf_label_size, i), 
            pdf_gap_shift=_get(pdf_gap_shift, i), 
            dn=_get(dn, i)
        )

def combine_rvs(rv_list):
    rvs = rv_list.copy()
    while len(rvs) > 2:
        n = math.floor(len(rvs) / 2)
        last_rv = rvs[-1]
        if len(rvs) > n * 2:
            is_odd = True
        else:
            is_odd = False
        rvs = [
            (rv_list[2 * i].update(rv_list[2 * i + 1])).normalize() 
            for i in range(n)
        ]
        if is_odd:
            rvs.append(last_rv)
    rv = rvs[0].update(rvs[1])
    return rv


