from .pdf_fitting_module import *


def C_XP(X, P):
    non_inf = (X > -1 * np.inf) & (X < 1 * np.inf)
    C = P.copy()
    C[non_inf] = cumulative_trapezoid(
        P[non_inf], X[non_inf], initial=0
    )
    C[non_inf] = C[non_inf] / C[non_inf].max()
    C[X == -1 * np.inf] = 0
    C[X == np.inf] = 1
    return C


def downsample(self, n_points):
    X = np.linspace(self.X.min(), self.X.max(), n_points)
    return self.match_X(X)


class CoreRandomVariable:
    
    def __init__(self, X, P, val=None, low=None, high=None):
        
        self.X = np.array(X)
        self.P = np.array(P)
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

    def new_kwargs(self):
        crv_code = CoreRandomVariable.__init__.__code__
        class_args = set(crv_code.co_varnames[1:crv_code.co_argcount])
        self_code = self.__init__.__code__
        self_args = set(self_code.co_varnames[1:self_code.co_argcount])
        new_args = self_args - class_args
        return {k:v for k, v in self.__dict__.items() if k in new_args}

    def __getitem__(self, a):
        if isinstance(a, slice):
            return self.__class__(
                self.X[a.start : a.stop : a.step],
                self.P[a.start : a.stop : a.step],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )
        elif np.array(a).shape == ():
            return np.interp(a, self.X, self.P)
        else:
            return self.__class__(
                self.X[a],
                self.P[a],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def C(self):
        return C_XP(self.X, self.P)

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
    
    def sample(self, n_samples, n_points=10000):
        X = np.linspace(self.X.min(), self.X.max(), n_points)
        P = np.interp(X, self.X, self.P)
        P = P / P.sum()
        return np.random.choice(X, p=P, size=n_samples)

    def function(self):
        def r_func(x):
            return np.interp(x, self.X, self.P)
        return r_func

    def normalize(self):
        integral = trapezoid(self.P, self.X)
        return self.__class__(
            self.X, self.P / integral, 
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def standardize(self):
        return self.__class__(
            self.X, self.P / self.P.max(),
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def match_X_of(self, other):
        return self.__class__(
            other.X, np.interp(other.X, self.X, self.P),
            val=self.val, low=self.low, high=self.high, 
            **self.new_kwargs()
        )

    def match_X(self, X, recalculate=False):
        if recalculate:
            return self.__class__(
                X, np.interp(X, self.X, self.P),
                val=None, low=None, high=None, 
                **self.new_kwargs()
            )
        else:
            return self.__class__(
                X, np.interp(X, self.X, self.P),
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def downsample(self, n_points):
        X = np.linspace(self.X.min(), self.X.max(), n_points)
        return self.match_X(X)

    def cut_below(self, c, recalculate=False):
        if recalculate:
            return self.__class__(
                self.X[self.X > c], self.P[self.X > c],
                val=None, low=None, high=None, **self.new_kwargs()
            )
        else:
            return self.__class__(
                self.X[self.X > c], self.P[self.X > c],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def cut_above(self, c, recalculate=False):
        if recalculate:
            return self.__class__(
                self.X[self.X < c], self.P[self.X < c],
                val=None, low=None, high=None, **self.new_kwargs()
            )
        else:
            return self.__class__(
                self.X[self.X < c], self.P[self.X < c],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def slice(self, Xmin, Xmax, recalculate=True):
        if recalculate:
            return self.__class__(
                self.X[(self.X >= Xmin) & (self.X <= Xmax)], 
                self.P[(self.X >= Xmin) & (self.X <= Xmax)],
                val=None, low=None, high=None, **self.new_kwargs()
            )
        else:
            return self.__class__(
                self.X[(self.X >= Xmin) & (self.X <= Xmax)], 
                self.P[(self.X >= Xmin) & (self.X <= Xmax)],
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )

    def update(self, other, log_space=False):
        if log_space:
            other_P = 10**np.interp(
                np.log10(self.X), np.log10(other.X), np.log10(other.P)
            )
        else:
            other_P = np.interp(self.X, other.X, other.P)
        kwargs = self.__dict__.copy()
        kwargs['val'] = None
        kwargs['low'] = None
        kwargs[''] = None
        return self.__class__(
            self.X, self.P * other_P, val=None, low=None, high=None,
            **self.new_kwargs()
        )

    def trim(self, precision=0.9999, recalculate=False):
        trim_max = self.percentile(precision)
        X_new = self.X[self.X < trim_max]
        P_new = self.P[self.X < trim_max]
        trim_min = self.percentile(1 - precision)
        if X_new[0] < 0:
            P_new = P_new[X_new > trim_min]
            X_new = X_new[X_new > trim_min]
        if recalculate:
            return self.__class__(
                X_new, P_new,
                val=None, low=None, high=None, **self.new_kwargs()
            )
        else:
            return self.__class__(
                X_new, P_new,
                val=self.val, low=self.low, high=self.high, 
                **self.new_kwargs()
            )


class BaseRandomVariable(CoreRandomVariable):

    def __init__(
        self, X, P, val=None, low=None, high=None, kind='median'
    ):
        super().__init__(X, P, val=val, low=low, high=high)
        self.kind = kind

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

    
    def as_kind(self, kind):
        if kind == self.kind:
            return self
        else:
            return self.__class__(
                self.X, self.P, low=None, val=None, high=None, kind=kind
            )

    def mean(self):
        return rv_mean_XP(self.X, self.P)

    def std(self):
        return rv_std_XP(self.X, self.P)

    def skewness(self):
        return rv_skewness_XP(self.X, self.P)

    def max(self):
        return self.X[np.argmax(self.P)]

    def mode(self):
        return self.X[np.argmax(self.P)]

    def median(self):
        return self.percentile(0.5)


    
def apply2rv(
    rv, f, kind=None, even_out=True, precision=1.0, 
    n_hide_near_roots=0
):
    X = rv.X
    PX = rv.P
    Y = f(X)
    is_valid = np.isfinite(Y) & ~np.isnan(Y)
    X, Y = X[is_valid], Y[is_valid]
    dYdX_sign = np.sign(np.diff(Y)[np.diff(Y) != 0])
    roots = np.where(np.diff(dYdX_sign) != 0)[0] + 1
    if roots.shape[0] > 0:
        PY = PX * np.abs(np.gradient(X, Y))
        n = n_hide_near_roots
        deletes = np.unique([
            np.arange(root - n, root + n + 1)
            for root in roots
        ])
        Y = np.delete(Y, deletes)
        PY = np.delete(PY, deletes)
        Ys, PYs = np.split(Y, roots), np.split(PY, roots)
        Y0 = np.sort(Y)
        PY0 = 0.0 * Y0
        for Yi, PYi in zip(Ys, PYs):
            if Yi[0] > Yi[-1]:
                Yi, PYi = np.flip(Yi), np.flip(PYi)
            in_range = (Y0 >= Yi.min()) & (Y0 <= Yi.max())
            Y0i = Y0[in_range]
            PY0[in_range] += np.interp(Y0i, Yi, PYi)
        Y, PY = Y0, PY0
    else:
        C = rv.C()[is_valid]
        PY = np.gradient(C, Y)
        Y = Y[~np.isnan(PY)]
        PY = PY[~np.isnan(PY)]
        if Y[0] > Y[-1]:
            PY = -1 * PY
            Y, PY = np.flip(Y), np.flip(PY)
    if kind is None:
        kind = rv.kind
    if even_out:
        Yd, PYd = Y[::3], PY[::3]
        CY = C_XP(Yd, PYd)
        Ymin = np.interp(1E-6, CY, Yd)
        Ymax = np.interp(1 - 1 / X.shape[0], CY, Yd)
        inc = np.gradient(CY, Yd).max() * 10000 / Y.shape[0] / precision
        Ylen = max(round(Ymax / inc), Y.shape[0])
        Y_even_spacing = np.linspace(
            Ymin, Ymax, Ylen, endpoint=True
        )
        PY_even_spacing = np.interp(Y_even_spacing, Y, PY)
        return rv.__class__(Y_even_spacing, PY_even_spacing, kind=kind)
    else:
        return rv.__class__(Y, PY, kind=kind)



def log__mul__(self, other, trim_tolerance=1E-10):
    log_self = self[self.X > 0].as_kind('mean').apply(
        np.log10, even_out=False
    ).pad_with_0s()
    log_other = other[other.X > 0].as_kind('mean').apply(
        np.log10, even_out=False
    ).pad_with_0s()
    return (log_self + log_other).apply(
        lambda x: 10**x, even_out=False
    ).as_kind(self.kind).trim(1 - trim_tolerance)



def log__truediv__(self, other, trim_tolerance=1E-10):
    log_self = self[self.X > 0].as_kind('mean').apply(
        np.log10, even_out=False
    ).pad_with_0s()
    log_other = other[other.X > 0].as_kind('mean').apply(
        np.log10, even_out=False
    ).pad_with_0s()
    return (log_self - log_other).apply(
        lambda x: 10**x, even_out=False
    ).as_kind(self.kind).trim(1 - trim_tolerance)



class MathRandomVariable(BaseRandomVariable):
    
    def __add__(self, other):
        if isinstance(other, MathRandomVariable):
            X_new_min = np.min([self.X.min(), other.X.min()])
            X_new_max = np.max([self.X.max(), other.X.max()])
            X_new_n = np.max([self.X.shape[0], other.X.shape[0]])
            X_new = np.linspace(X_new_min, X_new_max, X_new_n)
            self_P = np.interp(X_new, self.X, self.P)
            other_P = np.interp(X_new, other.X, other.P)
            conv_P = fftconvolve(self_P, other_P, mode='full')
            conv_n = conv_P.shape[0]
            conv_X = np.linspace(2 * X_new.min(), 2 * X_new.max(), conv_n)
            return self.__class__(conv_X, conv_P, kind=self.kind)
        elif other == 0:
            return self
        else:
            return self.__class__(
                self.X + other, self.P, val=self.val + other,
                low=self.low + other, high=self.high + other, 
                kind=self.kind
            )
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def pad_with_0s(self):
        X = np.append(self.X, self.X.max() + 1E-30)
        P = np.append(self.P, 0)
        if self.X.min() < 1E-30 and self.X.min() > 0:
            P[np.argmin(X)] = 0
        else:
            X = np.insert(X, 0, self.X.min() - 1E-30)
            P = np.insert(P, 0, 0)
        return self.__class__(
            X, P, val=self.val, low=self.low, high=self.high, kind=self.kind
        )
    
    def __mul__(self, other, integrate=False):
        if isinstance(other, MathRandomVariable):
            is_negative = (self.X.min() <= 0) or (other.X.min() <= 0)
            if integrate or is_negative:
                s = self.pad_with_0s()
                o = other.pad_with_0s()
                f1 = s.function()
                f2 = o.function()
                X1 = s.X
                X2 = o.X
                X1inc = (X1[-1] - X1[0]) / X1.shape[0]
                X2inc = (X2[-1] - X2[0]) / X2.shape[0]
                X1min, X1max = tuple(self.percentile(np.array([0.0001, 0.9999])))
                X2min, X2max = tuple(other.percentile(np.array([0.0001, 0.9999])))
                Xmin = X1min * X2min
                Xmax = X1max * X2max
                X1mean, X2mean = self.mean(), other.mean()
                Xlen = round(
                    (Xmax - Xmin) / max(X1inc, X2inc) / min(X1mean, X2mean)
                )
                Xlen = max(Xlen, X1.shape[0], X2.shape[0])
                Y = np.linspace(Xmin, Xmax, Xlen, endpoint=True)[:, np.newaxis]
                # Because Y has even spacing, we can use np.sum
                Py = (f1(X1) * f2(Y / X1) / np.abs(X1)).sum(axis=1)
                return self.__class__(Y.T[0], Py, kind=self.kind)
            else:
                return log__mul__(self, other)
        elif other == 0:
            return 0
        elif other < 0:
            return self.__class__(
                np.flip(self.X * other), np.flip(self.P), 
                val=self.val * other, high=self.low * other, 
                low=self.high * other, kind=self.kind
            )
        else:
            return self.__class__(
                self.X * other, self.P, val=self.val * other,
                low=self.low * other, high=self.high * other, 
                kind=self.kind
            )
    
    def __rmul__(self, other):
        if other == 0:
            return 0
        else:
            return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)
    
    def __rsub__(self, other):
        return (-1 * self) + other
        
    def __truediv__(self, other, integrate=False):
        if isinstance(other, MathRandomVariable):
            is_negative = (self.X.min() <= 0) or (other.X.min() <= 0)
            if integrate or is_negative:
                f1 = self.function()
                f2 = other.function()
                X1 = self.X
                X2 = other.X
                Y = np.linspace(
                    X1.min() / np.percentile(other.X, 99.99), 
                    X1.max() / np.percentile(other.X, 0.01), 
                    max(X1.shape[0], X2.shape[0]), 
                    endpoint=True
                )[:, np.newaxis]
                # Because Y has even spacing, we can use np.sum
                Py = (f1(Y * X2) * f2(X2) * np.abs(X2)).sum(axis=1)
                return self.__class__(Y.T[0], Py, kind=self.kind)
            else:
                return log__truediv__(self, other)
        elif other < 0:
            return self.__class__(
                self.X / other, self.P, val=self.val / other,
                high=self.low / other, low=self.high / other, 
                kind=self.kind
            )
        else:
            return self.__class__(
                self.X / other, self.P, val=self.val / other,
                low=self.low / other, high=self.high / other, 
                kind=self.kind
            )
        
    def __rtruediv__(self, other):
        if isinstance(other, MathRandomVariable):
            return other.__truediv__(self)
        elif other < 0:
            return self.__class__(
                other / self.X, self.P, val=other / self.val,
                high=other / self.low, low=other / self.high, 
                kind=self.kind
            )
        else:
            return self.apply(lambda x : other / x)
    
    def __rpow__(self, other):
        if isinstance(other, MathRandomVariable):
            raise ValueError(
                'The a**X operator is not for applying exponential '
                'functions to random variables.  It is for scaling '
                'random variables back out of log space with 10**X.  '
                'As a result, it cannot be applied to two random '
                'variables.'
            )
        else:
            if self.kind == 'linear':
                _kind = 'log'
            else:
                _kind = self.kind
            return self.__class__(
                other**self.X, self.P, val=other**self.val,
                low=other**self.low, high=other**self.high, 
                kind=_kind
            )

    def apply(self, f, kind=None, even_out=True):
        return apply2rv(self, f, kind=kind, even_out=even_out)

    def ten2the(self, kind=None, even_out=False):
        return self.apply(
            lambda x: 10**x, kind=kind, even_out=even_out
        )

    def scale(self, f, recalculate_bounds=True):
        X, P = f(self.X), self.P
        if X[0] > X[-1]:
            X, P = np.flip(X), np.flip(P)
        if recalculate_bounds:
            return self.__class__(
                X, P, low=None, val=None, high=None, kind=self.kind
            )
        else:
            return self.__class__(
                X, P, kind=self.kind,
                val=f(self.val), low=f(self.low), high=f(self.high)
            )

    def log(self, recalculate_bounds=False):
        rv_log = self[self.X > 0].scale(
            np.log10, recalculate_bounds=recalculate_bounds
        )
        if self.kind.lower() in {'log', 'auto log'}:
            rv_log.kind = 'linear'
        return rv_log


