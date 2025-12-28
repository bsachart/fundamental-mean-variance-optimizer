import pytest
import polars as pl
import numpy as np
from src.engine.risk import calculate_covariance, RiskModel


@pytest.fixture
def mock_prices():
    """Create a deterministic synthetic price series."""
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "A": [100.0, 101.0, 102.0, 103.0, 104.0],
        "B": [50.0, 50.5, 51.0, 51.5, 50.0],
    }
    return pl.DataFrame(data)


def test_historical_covariance_scaling(mock_prices):
    """Test that historical covariance scales with annualization_factor."""
    cov_1 = calculate_covariance(
        mock_prices, risk_model=RiskModel.HISTORICAL, annualization_factor=1
    )
    cov_252 = calculate_covariance(
        mock_prices, risk_model=RiskModel.HISTORICAL, annualization_factor=252
    )
    np.testing.assert_allclose(cov_252, cov_1 * 252)


def test_hybrid_covariance_structure(mock_prices):
    """Test that hybrid model uses Implied Vols for diagonal."""
    implied_vols = np.array([0.20, 0.30])
    cov = calculate_covariance(
        mock_prices, risk_model=RiskModel.FORWARD_LOOKING, implied_vols=implied_vols
    )
    diagonals = np.diag(cov)
    expected_diagonals = implied_vols**2
    np.testing.assert_allclose(diagonals, expected_diagonals)

    # Check off-diagonal structure (correlation preservation)
    ticker_data = mock_prices.select(["A", "B"]).to_numpy()
    log_rets = np.diff(np.log(ticker_data), axis=0)
    actual_corr = np.corrcoef(log_rets, rowvar=False)[0, 1]
    expected_cov_ab = actual_corr * 0.20 * 0.30
    assert np.isclose(cov[0, 1], expected_cov_ab)


def test_error_handling(mock_prices):
    """Test robust error handling for missing parameters."""
    # Missing implied_vols for FORWARD_LOOKING
    with pytest.raises(ValueError, match="implied_vols is required"):
        calculate_covariance(mock_prices, risk_model=RiskModel.FORWARD_LOOKING)

    # Wrong shape implied_vols
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_covariance(
            mock_prices,
            risk_model=RiskModel.FORWARD_LOOKING,
            implied_vols=np.array([0.1, 0.2, 0.3]),
        )

    # Missing annualization_factor for HISTORICAL
    with pytest.raises(ValueError, match="annualization_factor is required"):
        calculate_covariance(mock_prices, risk_model=RiskModel.HISTORICAL)


def test_single_asset():
    """Test with single asset (1x1 covariance matrix)."""
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "A": [100.0, 101.0, 102.0],
    }
    prices = pl.DataFrame(data)
    cov = calculate_covariance(
        prices, risk_model=RiskModel.HISTORICAL, annualization_factor=252
    )
    assert cov.shape == (1, 1)
    assert cov[0, 0] > 0


def test_zero_variance_asset():
    """Test handling of constant price (zero variance)."""
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "A": [100.0, 101.0, 102.0, 103.0],
        "B": [50.0, 50.0, 50.0, 50.0],  # Constant
    }
    prices = pl.DataFrame(data)

    # Historical: should be 0.0
    cov_hist = calculate_covariance(
        prices, risk_model=RiskModel.HISTORICAL, annualization_factor=252
    )
    assert cov_hist[1, 1] == 0.0

    # Forward-Looking: should be implied vol squared
    implied_vols = np.array([0.20, 0.30])
    cov_fwd = calculate_covariance(
        prices,
        risk_model=RiskModel.FORWARD_LOOKING,
        implied_vols=implied_vols,
    )
    assert cov_fwd[1, 1] == 0.30**2


def test_perfect_correlation():
    """Test assets with perfect positive correlation."""
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "A": [100.0, 110.0, 105.0, 115.0],  # Returns vary
        "B": [50.0, 55.0, 52.5, 57.5],  # Same % moves as A
    }
    prices = pl.DataFrame(data)
    implied_vols = np.array([0.20, 0.30])

    cov = calculate_covariance(
        prices, risk_model=RiskModel.FORWARD_LOOKING, implied_vols=implied_vols
    )

    expected_cov = 1.0 * 0.20 * 0.30
    # Correlation should be 1.0
    assert np.isclose(cov[0, 1], expected_cov)


def test_symmetric_matrix(mock_prices):
    """Covariance matrix should be symmetric."""
    cov = calculate_covariance(
        mock_prices, risk_model=RiskModel.HISTORICAL, annualization_factor=252
    )
    assert np.allclose(cov, cov.T)
