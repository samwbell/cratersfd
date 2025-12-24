from .random_variable_module import *


def fit_pf(X, Y):
    def f(D):
        return np.interp(D, X, Y)
    return f


def linear2loglog_pf(pf):
    def loglog_pf(logD):
        return np.log10(pf(10**logD))
    return loglog_pf


def loglog2linear_pf(pf):
    def linear_pf(d):
        return 10**pf(np.log10(d))
    return linear_pf


def cumulative2differential_pf(
    c_pf, dmin=0.005, dmax=1E4, n_points=10000
):
    D = np.logspace(np.log10(dmin), np.log10(dmax), n_points)
    differential_array = -1 * np.gradient(c_pf(D), D)
    def d_pf(d):
        return np.interp(d, D, differential_array)
    return d_pf


def differential2cumulative_pf(
    d_pf, dmin=0.005, dmax=1E4, n_points=10000
):
    D = np.logspace(np.log10(dmin), np.log10(dmax), n_points)
    fD = np.flip(D)
    cumulative_array = -1 * np.flip(
        cumulative_trapezoid(d_pf(fD), fD, initial=0)
    )
    def c_pf(d):
        return np.interp(d, D, cumulative_array)
    return c_pf


def differential2R_pf(
    d_pf, dmin=0.005, dmax=1E4, n_points=10000
):
    def R_pf(d):
        return d_pf(d) * d**3
    return R_pf
    

def R2differential_pf(
    R_pf, dmin=0.005, dmax=1E4, n_points=10000
):
    def d_pf(d):
        return R_pf(d) / d**3
    return d_pf


def get_saturated_pf(X, Y, plot_type='differential', at=None):
    if plot_type == 'R':
        r_pf = fit_pf(X, Y)
    elif plot_type == 'differential':
        d_pf = fit_pf(X, Y)
        r_pf = differential2R_pf(d_pf)
    else:
        raise ValueError(
            'plot_type must either be \'differential\' or \'R\''
        )
    if at is None:
        peak_D = X[np.argmax(r_pf(X))]
        peak_R = np.max(r_pf(X))
    else:
        peak_D = at
        peak_R = r_pf(at)
    def s_pf(D):
        return np.piecewise(
            D, [D < peak_D, D >= peak_D],
            [r_pf(D[D < peak_D]), peak_R]
        )
    if plot_type == 'R':
        return s_pf
    else:
        return R2differential_pf(s_pf)


def linear_pf(N1=0.001, slope=-2):
    def out_f(d):
        return 10**(slope * np.log10(d) + np.log10(N1))
    return out_f


def linear_pf_R(alpha=2, D=1, R=0.1):
    def out_f(d):
        m = 2 - alpha
        x = np.log10(d)
        x1 = np.log10(D)
        y1 = np.log10(R)
        return 10**(m * (x - x1) + y1)
    return out_f


def loglog_linear_pf(N1=0.001, slope=-2):
    def out_f(logd):
        return slope * logd + np.log10(N1)
    return out_f

        
def polynomial_pf(D, coefficients):
    logD = np.log10(D)
    a_n = np.array(coefficients)
    n_max = a_n.shape[0]
    n = np.arange(n_max)
    logD_matrix = np.tile(logD, (n_max, 1)).T
    N1 = 10**np.sum(a_n * logD_matrix**n, axis=1)
    if N1.shape[0] == 1:
        N1 = float(N1[0])
    return N1


def polynomial_pf_dif(D, coefficients):
    logD = np.log10(D)
    a_n = np.array(coefficients)
    n_max = a_n.shape[0]
    n = np.arange(1, n_max)
    p = polynomial_pf(D, coefficients)
    logD_matrix = np.tile(logD, (n_max - 1, 1)).T
    summation = np.sum(a_n[1:] * n * logD_matrix**(n - 1), axis=1)
    return -1 * p / D * summation


def polynomial_pf_R(D, coefficients):
    return polynomial_pf_dif(D, coefficients) * D**3


npf_new_coefficients = np.array([
    -3.0876, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058, 0.019977,
    0.086850, -0.005874, -0.006809, 8.25*10**-4, 5.54*10**-5
])


def npf_new(D):
    return polynomial_pf(D, npf_new_coefficients)
    

def npf_new_dif(D):
    return polynomial_pf_dif(D, npf_new_coefficients)


def npf_new_R(D):
    return polynomial_pf_R(D, npf_new_coefficients)


def npf_new_loglog(logD):
    return np.log10(npf_new(10**logD))


_logD = np.linspace(np.log10(0.005), np.log10(2500), 20000)
_logRho = npf_new_loglog(_logD)
_npf_slope_array = np.gradient(_logRho, _logD)
def npf_slope(d):
    return np.interp(np.log10(d), _logD, _npf_slope_array)
_logDifRho = np.log10(npf_new_dif(10**_logD))
_npf_alpha_array = np.gradient(_logDifRho, _logD)
def npf_alpha(d):
    return -1 * np.interp(np.log10(d), _logD, _npf_alpha_array) - 1


def npf_error(D):
    m0 = 0.4 / (np.log10(0.8) - np.log10(0.1))
    m1 = 0.1 / (0 - np.log10(0.8))
    m2 = 0.1 / np.log10(3)
    m3 = 0.65 / (np.log10(75) - np.log10(3))
    D = np.array(D).astype('float')
    return np.piecewise(
        D,
        [
            D <= 0.8,
            (D > 0.8) & (D <= 1.0),
            (D > 1) & (D <= 3.0),
            D > 3.0
        ],
        [
            0.1 + m0 * (np.log10(0.8) - np.log10(D)),
            m1 * (0 - np.log10(D)),
            m2 * np.log10(D),
            0.1 + m3 * (np.log10(D) - np.log10(3))
        ]
    )


npf_mars_coefficients = np.array([
    -3.384, -3.197, 1.257, 0.7915, -0.4861, -0.3630, 0.1016,
    6.756E-2, -1.181E-2, -4.753E-3, 6.233E-4, 5.805E-5
])


def npf_mars(D):
    return polynomial_pf(D, npf_mars_coefficients)


def relative_npf_error(d1, d2):
    return subtract_lognormal_ps(npf_error(d1), npf_error(d2))


def geometric_sat(D):
    return 0.385 * (D / 2)**-2


def hartmann84_sat(D):
    return 10**(-1.83 * np.log10(D) - 1.33)


def hartmann84_sat_D(age, Ds, pf=npf_new_loglog):
    d = np.flip(Ds)
    diff = np.log10(age) + pf(np.log10(d)) - np.log10(hartmann84_sat(d))
    return np.interp(0, diff, d)


# The Neukum Chronology Function.
def ncf(t):
    return 5.44E-14 *(np.exp(6.93 * t) - 1) + 8.38E-4 * t

def dncf_dt(t):
    return 5.44E-14 * 6.93 * np.exp(6.93 * t) + 8.38E-4

# This is an object used in the calculation of ncf_inv.
class ncf_model():
    def __init__(self, nseg=10000):
        self.nseg_pts = nseg
        self.T = np.linspace(0, 5, nseg)
        self.N1 = ncf(self.T)
    def inv(self, N1):
        return np.interp(N1, self.N1, self.T)
    
_ncf_model = ncf_model(nseg=1000000)
    
def ncf_inv(N1, ncf_model=_ncf_model):
    return ncf_model.inv(N1)
    

def ncf_mars(t):
    return 2.68E-14 *(np.exp(6.93 * t) - 1) + 4.13E-4 * t

class ncf_mars_model():
    def __init__(self, nseg=10000):
        self.nseg_pts = nseg
        self.T = np.linspace(0, 5, nseg)
        self.N1 = ncf_mars(self.T)
    def inv(self, N1):
        return np.interp(N1, self.N1, self.T)
    
_ncf_mars_model = ncf_mars_model(nseg=1000000)
    
def ncf_mars_inv(N1, ncf_mars_model=_ncf_mars_model):
    return ncf_mars_model.inv(N1)



def synth_pf(d_km):
    return 10**(-3 - 2*np.log10(d_km))
    

def synth_cf_inv(n1):
    return n1/synth_pf(1)
    

