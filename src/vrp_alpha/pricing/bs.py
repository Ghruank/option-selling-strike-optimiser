"""Vectorised Black-Scholes-Merton for European index options (continuous yield q).

All functions broadcast over numpy arrays. `is_call` is a boolean (array).
T is in years (calendar days / 365).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def d1_d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_t
    return d1, d1 - vol_t


def price(S, K, T, r, q, sigma, is_call):
    """Option value. At T<=0 returns intrinsic value."""
    S, K, T, sigma, is_call = np.broadcast_arrays(*map(np.asarray, (S, K, T, sigma, is_call)))
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    live = T > 1e-10
    Tl = np.where(live, T, 1.0)
    d1, d2 = d1_d2(S, K, Tl, r, q, sigma)
    df_r, df_q = np.exp(-r * Tl), np.exp(-q * Tl)
    call = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    put = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    out = np.where(live, np.where(is_call, call, put), intrinsic)
    return out[()] if out.ndim == 0 else out


def greeks(S, K, T, r, q, sigma, is_call) -> dict[str, np.ndarray]:
    """delta, gamma, vega (per 1.00 vol), theta (per year) for a long option."""
    S, K, T, sigma, is_call = np.broadcast_arrays(*map(np.asarray, (S, K, T, sigma, is_call)))
    T = np.maximum(T, 1e-8)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = _npdf(d1)
    delta = np.where(is_call, df_q * ndtr(d1), -df_q * ndtr(-d1))
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common = -S * df_q * pdf * sigma / (2 * np.sqrt(T))
    theta_c = common - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
    theta_p = common + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
    theta = np.where(is_call, theta_c, theta_p)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def prob_itm_rn(S, K, T, r, q, sigma, is_call):
    """Risk-neutral probability of finishing in the money, N(d2) / N(-d2)."""
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    return np.where(is_call, ndtr(d2), ndtr(-d2))


def implied_vol(px, S, K, T, r, q, is_call, lo=1e-3, hi=5.0, iters=100):
    """Vectorised bisection; NaN where the price violates no-arbitrage bounds."""
    px, S, K, T, is_call = np.broadcast_arrays(*map(np.asarray, (px, S, K, T, is_call)))
    px = px.astype(float)
    lo_a = np.full(px.shape, lo)
    hi_a = np.full(px.shape, hi)
    p_lo = price(S, K, T, r, q, lo_a, is_call)
    p_hi = price(S, K, T, r, q, hi_a, is_call)
    valid = (px > p_lo) & (px < p_hi)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        above = price(S, K, T, r, q, mid, is_call) > px
        hi_a = np.where(above, mid, hi_a)
        lo_a = np.where(above, lo_a, mid)
    iv = 0.5 * (lo_a + hi_a)
    out = np.where(valid, iv, np.nan)
    return out[()] if out.ndim == 0 else out


def strike_from_delta(S, T, r, q, sigma, abs_delta, is_call):
    """Strike with |delta| = abs_delta for a flat vol sigma."""
    x = ndtri(np.asarray(abs_delta) * np.exp(q * T))
    d1 = np.where(is_call, x, -x)
    return S * np.exp(-d1 * sigma * np.sqrt(T) + (r - q + 0.5 * sigma**2) * T)
