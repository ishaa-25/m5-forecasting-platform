import numpy as np
import pandas as pd
import pytest

from m5forecast.reconciliation.hierarchy import (
    build_summing_matrix,
    reconcile_mint_shrink,
    reconcile_ols,
)


@pytest.fixture
def mock_bottom_df():
    """Create a minimal hierarchical bottom-level dataframe representing M5 hierarchy."""
    data = {
        "item_id": ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOBBIES_2"],
        "dept_id": ["FOODS", "FOODS", "HOBBIES", "HOBBIES"],
        "cat_id": ["FOODS_CAT", "FOODS_CAT", "HOBBIES_CAT", "HOBBIES_CAT"],
        "store_id": ["CA_1", "CA_1", "CA_1", "CA_1"],
        "state_id": ["CA", "CA", "CA", "CA"],
    }
    return pd.DataFrame(data)


def test_build_summing_matrix(mock_bottom_df):
    id_cols = ["item_id", "store_id"]
    S, all_ids = build_summing_matrix(mock_bottom_df, id_cols=id_cols)

    n_bottom = len(mock_bottom_df)
    # Check dimensions: S should have n_bottom columns
    assert S.shape[1] == n_bottom
    # S matrix should consist of 0s and 1s
    assert np.all((S == 0) | (S == 1))

    # Test summing property: y_all = S @ y_bottom
    y_bottom = np.array([10.0, 20.0, 30.0, 40.0])
    y_all = S @ y_bottom

    # Total row (first element) should equal sum of all bottom items (100.0)
    assert y_all[0] == 100.0


def test_reconcile_ols_coherence(mock_bottom_df):
    id_cols = ["item_id", "store_id"]
    S, all_ids = build_summing_matrix(mock_bottom_df, id_cols=id_cols)
    n_all, n_bottom = S.shape
    horizon = 5

    # Simulate incoherent base forecasts (random noise around true values)
    np.random.seed(42)
    base_forecasts = np.random.uniform(5, 50, size=(n_all, horizon))

    # Apply OLS reconciliation
    reconciled = reconcile_ols(base_forecasts=base_forecasts, S=S)

    # Verify output shape
    assert reconciled.shape == (n_all, horizon)

    # Verify coherence property: S @ bottom_reconciled == all_reconciled
    bottom_reconciled = reconciled[-n_bottom:, :]
    coherent_all = S @ bottom_reconciled

    # Maximum difference across all levels and horizon steps should be effectively 0
    max_incoherence_error = np.max(np.abs(reconciled - coherent_all))
    assert max_incoherence_error < 1e-10


def test_reconcile_mint_shrink_coherence(mock_bottom_df):
    id_cols = ["item_id", "store_id"]
    S, all_ids = build_summing_matrix(mock_bottom_df, id_cols=id_cols)
    n_all, n_bottom = S.shape
    horizon = 7
    n_history = 30

    # Simulate random in-sample residuals and base forecasts
    np.random.seed(123)
    residuals = np.random.normal(loc=0, scale=2.0, size=(n_all, n_history))
    base_forecasts = np.random.uniform(10, 100, size=(n_all, horizon))

    # Apply MinT shrinkage reconciliation
    reconciled = reconcile_mint_shrink(
        base_forecasts=base_forecasts, S=S, residuals=residuals
    )

    # Verify output shape
    assert reconciled.shape == (n_all, horizon)

    # Verify mathematical coherence: S @ y_bottom == y_all
    bottom_reconciled = reconciled[-n_bottom:, :]
    coherent_all = S @ bottom_reconciled

    max_incoherence_error = np.max(np.abs(reconciled - coherent_all))
    assert max_incoherence_error < 1e-10