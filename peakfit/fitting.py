"""Fitting wrappers using scipy and optionally lmfit."""
from typing import Dict, Any, Tuple
import numpy as np

from .model import model as model_func


def _build_free_params(config: Dict[str, Any]):
    free_names = []
    p0 = []
    lower = []
    upper = []
    for name, info in config.items():
        vary = bool(info.get("vary", True))
        if vary:
            free_names.append(name)
            p0.append(float(info.get("value", 0.0)))
            lower.append(info.get("min", -np.inf))
            upper.append(info.get("max", np.inf))
    return free_names, np.array(p0), (np.array(lower, dtype=float), np.array(upper, dtype=float))


def _params_from_vector(free_names, pvec, config: Dict[str, Any]):
    params = {}
    j = 0
    for name, info in config.items():
        if info.get("vary", True):
            params[name] = float(pvec[j])
            j += 1
        else:
            params[name] = float(info.get("value", 0.0))
    return params


def fit_with_scipy(iw, y, param_config: Dict[str, Any], integrator_opts=None,
                   curvefit_opts=None, progress_callback=None):
    """Fit using scipy.optimize.curve_fit.

    Returns: (params_dict, errs_dict, popt, pcov, y_model, r2, converged, message)
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as e:
        raise ImportError("scipy is required for fitting with scipy.curve_fit") from e

    iw = np.asarray(iw, dtype=float)
    y = np.asarray(y, dtype=float)
    integrator_opts = integrator_opts or {}
    curvefit_opts = curvefit_opts or {}

    free_names, p0, bounds = _build_free_params(param_config)
    if len(free_names) == 0:
        params_full = {name: float(info.get("value", 0.0)) for name, info in param_config.items()}
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_opts)
        # compute R^2
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        # No fitting performed; treat as converged
        return params_full, errs_full, None, None, y_model, r2, True, "no free parameters"

    call_count = {'n': 0}
    def fitfunc(iw_vals, *pvec):
        call_count['n'] += 1
        if progress_callback is not None:
            try:
                cont = bool(progress_callback(call_count['n'], pvec))
            except Exception:
                cont = True
            if not cont:
                raise RuntimeError('fitting cancelled by user')
        params_full = _params_from_vector(free_names, pvec, param_config)
        return model_func(iw_vals, params_full, **integrator_opts)

    converged = False
    message = ""
    try:
        popt, pcov = curve_fit(fitfunc, iw, y, p0=p0, bounds=bounds, **curvefit_opts)
        params_full = _params_from_vector(free_names, popt, param_config)
        converged = True
        message = "ok"
        # If covariance is not usable, mark as not converged
        try:
            if pcov is None or not np.all(np.isfinite(pcov)):
                converged = False
                message = "covariance could not be estimated or contains non-finite values"
        except Exception:
            # leave covariance-check best-effort
            pass
    except Exception as e:
        # fit failed: return initial parameter guess and an informative message
        converged = False
        message = f"curve_fit failed: {e}"
        popt = p0
        pcov = None
        params_full = _params_from_vector(free_names, popt, param_config)
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_opts)
        try:
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        except Exception:
            r2 = float('nan')
        return params_full, errs_full, popt, pcov, y_model, r2, converged, message

    # compute parameter uncertainties (standard deviations)
    errs_full = {name: None for name in param_config.keys()}
    try:
        if pcov is not None:
            # extract std for free parameters
            perr = np.sqrt(np.diag(pcov))
            for j, name in enumerate(free_names):
                errs_full[name] = float(perr[j])
    except Exception:
        # leave errs as None if computation fails
        pass

    y_model = model_func(iw, params_full, **integrator_opts)

    # compute R^2
    try:
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        r2 = float('nan')

    return params_full, errs_full, popt, pcov, y_model, r2, converged, message


def fit_with_lmfit(iw, y, param_config: Dict[str, Any], integrator_opts=None,
                   minimizer_opts=None, progress_callback=None):
    """Fit using lmfit if available. Returns (params_dict, errs_dict, result, y_model, r2, converged, message)."""
    try:
        import lmfit
    except Exception as e:
        raise ImportError("lmfit is required for this backend; install it or use scipy backend") from e

    iw = np.asarray(iw, dtype=float)
    y = np.asarray(y, dtype=float)
    integrator_opts = integrator_opts or {}
    minimizer_opts = minimizer_opts or {}

    params = lmfit.Parameters()
    for name, info in param_config.items():
        kwargs = {}
        if "min" in info:
            kwargs["min"] = info["min"]
        if "max" in info:
            kwargs["max"] = info["max"]
        params.add(name, value=info.get("value", 0.0), vary=info.get("vary", True), **kwargs)

    call_count = {'n': 0}
    def resid(p):
        call_count['n'] += 1
        if progress_callback is not None:
            try:
                cont = bool(progress_callback(call_count['n'], None))
            except Exception:
                cont = True
            if not cont:
                raise RuntimeError('fitting cancelled by user')
        pvals = {n: p[n].value for n in p}
        y_model = model_func(iw, pvals, **integrator_opts)
        return y - y_model

    minimizer = lmfit.Minimizer(resid, params)
    result = minimizer.minimize(**minimizer_opts)
    fitted = {n: float(result.params[n].value) for n in result.params}
    errs = {n: (float(result.params[n].stderr) if result.params[n].stderr is not None else None) for n in result.params}
    # lmfit returns a Result object with .success and .message attributes
    converged = bool(getattr(result, 'success', False))
    message = str(getattr(result, 'message', ''))
    y_model = model_func(iw, fitted, **integrator_opts)
    try:
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        r2 = float('nan')
    return fitted, errs, result, y_model, r2, converged, message
