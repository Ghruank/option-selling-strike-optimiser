import numpy as np

from vrp_alpha.pricing import bs


def test_put_call_parity():
    S, K, T, r, q, v = 100.0, np.array([80, 100, 120.0]), 0.25, 0.04, 0.015, 0.22
    c = bs.price(S, K, T, r, q, v, True)
    p = bs.price(S, K, T, r, q, v, False)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_known_value():
    # Hull, Options Futures & Other Derivatives: S=42 K=40 r=10% T=0.5 vol=20% -> call 4.76, put 0.81
    assert abs(bs.price(42, 40, 0.5, 0.1, 0.0, 0.2, True) - 4.76) < 0.01
    assert abs(bs.price(42, 40, 0.5, 0.1, 0.0, 0.2, False) - 0.81) < 0.01


def test_implied_vol_roundtrip():
    K = np.linspace(70, 130, 13)
    vol = 0.15 + 0.002 * np.abs(K - 100)
    is_call = K > 100
    px = bs.price(100, K, 0.1, 0.03, 0.01, vol, is_call)
    np.testing.assert_allclose(bs.implied_vol(px, 100, K, 0.1, 0.03, 0.01, is_call), vol, atol=1e-6)


def test_strike_from_delta():
    for is_call in (True, False):
        K = bs.strike_from_delta(100, 0.08, 0.03, 0.01, 0.2, 0.25, is_call)
        d = bs.greeks(100, K, 0.08, 0.03, 0.01, 0.2, is_call)["delta"]
        assert abs(abs(d) - 0.25) < 1e-9


def test_expiry_intrinsic():
    assert bs.price(90, 100, 0.0, 0.03, 0.0, 0.2, False) == 10.0
    assert bs.price(90, 100, 0.0, 0.03, 0.0, 0.2, True) == 0.0
