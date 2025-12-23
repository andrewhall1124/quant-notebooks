import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Returns and Portfolio Optimization
    In this notebook we will illustrate the foundational components of portfolio management and discuss the following topics:
    - Return distributions (mean and volatility)
    - Sharpe ratios
    - Single stock portfolios
    - Minimum variance portfolios
    - Tangent portfolios
    - The Capital Allocation Line
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 0: Setup

    First we need to import the necessary packages and create a random number generator.
    """)
    return


@app.cell
def _():
    import polars as pl
    import yfinance as yf
    import numpy as np
    import altair as alt
    import cvxpy as cp
    return alt, cp, np, pl, yf


@app.cell
def _(np):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    return (rng,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1: Distributions of Returns

    - We generally characterize stock return performance in terms of mean and standard deviation.
    - We will illustrate this as a synthetic security with mean 15bps and standard deviation 150bps (daily returns).
    """)
    return


@app.cell
def _():
    mean = 0.0015  # 15bs per day
    stdev = 0.0150  # 150bps per day
    n = 252 * 1
    return mean, n, stdev


@app.cell
def _(mean, n, np, pl, rng, stdev):
    # Sample returns from normal distribution
    returns_np = np.array([rng.normal() * stdev + mean for _ in range(n)])

    # Create a dataframe of returns and calculate cumulative returns
    returns_df = (
        pl.DataFrame({"return": returns_np})
        .with_row_index()
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .sub(1)
            .mul(100)
            .alias("cumulative_return")
        )
    )
    return returns_df, returns_np


@app.cell
def _(mo):
    mo.md(r"""
    ### Daily Returns Distribution of Synthetic Asset
    """)
    return


@app.cell
def _(alt, returns_df):
    (
        alt.Chart(returns_df)
        .mark_bar()
        .encode(
            x=alt.X("return", title="Return", bin=alt.Bin(maxbins=100)), y="count()"
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cumulative Performance of Synthetic Asset
    - We can annualize daily returns by multiplying by 252
    - We can annualize daily volatilities by multiplying by the square root of 252
    - The foundational measure of performance is the Sharpe Ratio which is equivalent to:
    """)
    return


@app.cell
def _(alt, returns_df):
    (
        alt.Chart(returns_df)
        .mark_line()
        .encode(
            x=alt.X("index", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Return"),
        )
    )
    return


@app.cell
def _(np, returns_np):
    annual_return = returns_np.mean() * 252
    annual_volatility = returns_np.std() * np.sqrt(252)
    annual_sharpe = annual_return / annual_volatility

    print(f"Return: {annual_return:.2%}")
    print(f"Volatility: {annual_volatility:.2%}")
    print(f"Sharpe: {annual_sharpe:.2f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### AAPL Stock Returns
    To prove our point we will illustrate the same process but use historical prices for AAPL from 2024-01-01 to 2024-12-31
    """)
    return


@app.cell
def _(pl, yf):
    aapl_df = (
        pl.DataFrame(
            yf.download(tickers=["AAPL"], start="2024-01-01", end="2024-12-31")
            .stack(future_stack=True)
            .reset_index()
        )
        .rename(
            {
                "Date": "date",
                "Ticker": "ticker",
                "Close": "close",
            }
        )
        .select("date", "ticker", "close")
        .with_columns(pl.col("close").pct_change().fill_null(0).alias("return"))
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .sub(1)
            .mul(100)
            .fill_null(0)
            .alias("cumulative_return")
        )
    )
    return (aapl_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Daily Returns Distribution of AAPL
    """)
    return


@app.cell
def _(aapl_df, alt):
    (
        alt.Chart(aapl_df)
        .mark_bar()
        .encode(
            x=alt.X("return", title="Return", bin=alt.Bin(maxbins=100)), y="count()"
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cumulative Performance of AAPL
    """)
    return


@app.cell
def _(aapl_df, alt):
    (
        alt.Chart(aapl_df)
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Return"),
        )
    )
    return


@app.cell
def _(aapl_df, np):
    aapl_returns_np = aapl_df["return"].to_numpy()

    annual_aapl_return = aapl_returns_np.mean() * 252
    annual_aapl_volatility = aapl_returns_np.std() * np.sqrt(252)
    annual_aapl_sharpe = annual_aapl_return / annual_aapl_volatility

    print(f"Return: {annual_aapl_return:.2%}")
    print(f"Volatility: {annual_aapl_volatility:.2%}")
    print(f"Sharpe: {annual_aapl_sharpe:.2f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2: Portfolios
    - We will now look at how combining assets into portfolios allows us to create new assets.
    - For our examples we will considers that included AAPL, IBM, WMT, and VZ.
    - We will compare each portfolio in terms of it's annualized return and volatility.
    - This will involve plotting each portfolio in risk and return space.
    """)
    return


@app.cell
def _(pl, yf):
    tickers = ["AAPL", "IBM", "WMT", "VZ"]
    stocks_df = (
        pl.DataFrame(
            yf.download(tickers=tickers, start="2024-01-01", end="2024-12-31")
            .stack(future_stack=True)
            .reset_index()
        )
        .rename(
            {
                "Date": "date",
                "Ticker": "ticker",
                "Close": "close",
            }
        )
        .select("date", "ticker", "close")
        .sort("date", "ticker")
        .with_columns(
            pl.col("close").pct_change().fill_null(0).over("ticker").alias("return")
        )
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .sub(1)
            .mul(100)
            .fill_null(0)
            .over("ticker")
            .alias("cumulative_return")
        )
    )
    return stocks_df, tickers


@app.cell
def _(mo):
    mo.md(r"""
    ### Cumulative Performance of Single Stock Portfolios
    """)
    return


@app.cell
def _(alt, stocks_df):
    (
        alt.Chart(stocks_df)
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Return"),
            color=alt.Color("ticker", title="Ticker"),
        )
    )
    return


@app.cell
def _(np, pl, stocks_df):
    # Compute performance of single stock portfolios
    single_stock_portfolios = (
        stocks_df.group_by("ticker")
        .agg(
            pl.col("return").mean().mul(252).alias("mean_return"),
            pl.col("return").std().mul(np.sqrt(252)).alias("volatility"),
        )
        .sort("ticker")
    )
    return (single_stock_portfolios,)


@app.cell
def _(alt, pl):
    def plot_portfolios(portfolios: list[pl.DataFrame]) -> None:
        return (
            alt.Chart(pl.concat(portfolios))
            .mark_point(filled=True, size=100)
            .encode(
                x=alt.X("volatility", title="Volatility"),
                y=alt.Y("mean_return", title="Mean Return"),
                color=alt.Color("ticker", title="Ticker"),
            )
            .transform_calculate(
                mean_return="datum.mean_return * 100",
                volatility="datum.volatility * 100",
            )
        )
    return (plot_portfolios,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Single Stock Portfolios
    """)
    return


@app.cell
def _(plot_portfolios, single_stock_portfolios):
    plot_portfolios([single_stock_portfolios])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Expected Returns and Covariances
    - In order to compute the portfolio level return and volatility we need two components.
    - Expected Returns Vector: we will use the average of historical returns as a naive forecast of future returns.
    - Covariance Matrix: we will use the historical covariances of returns as a naive forecast of future covariances.
    """)
    return


@app.cell
def _(np, pl, stocks_df):
    # Get expected return vector
    expected_returns = (
        stocks_df.group_by("ticker").agg(pl.col("return").mean())["return"].to_numpy()
    )

    # Get covariance matrix
    covariance_matrix = np.cov(
        stocks_df.pivot(on="ticker", index="date", values="return")
        .drop("date")
        .to_numpy()
        .T
    )
    return covariance_matrix, expected_returns


@app.cell
def _(mo):
    mo.md(r"""
    ### Random Portfolios
    - We will illustrate some of the potential portfolios we can create from our selected universe using random portfolios weight.
    - One constraint we will impose is that our portfolios are fully invested meaning that the weights sum to one.
    """)
    return


@app.cell
def _(covariance_matrix, expected_returns, np, pl, rng, tickers):
    # Sample a bunch of random portfolio weights
    weights_list = rng.uniform(0, 1, (100, len(tickers)))

    # Enforce that they sum to 1
    weights_list = weights_list / weights_list.sum(axis=1, keepdims=True)

    # Generate random portfolios
    random_portfolios_list = []
    for _weights in weights_list:
        _portfolio_return = _weights.T @ expected_returns * 252
        _portfolio_volatility = np.sqrt(
            _weights.T @ covariance_matrix @ _weights
        ) * np.sqrt(252)

        random_portfolios_list.append(
            {
                "ticker": "Random",
                "mean_return": _portfolio_return,
                "volatility": _portfolio_volatility,
            }
        )

    random_portfolios = pl.DataFrame(random_portfolios_list)
    return random_portfolios, weights_list


@app.cell
def _(plot_portfolios, random_portfolios, single_stock_portfolios):
    plot_portfolios([single_stock_portfolios, random_portfolios])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Minimum Variance Portfolios
    - Now we will optimize each of our random portfolios by running the expected returns vector and covariance matrix through an optimizer.
    - Our object is to minimize the variance of the portfolio, and our constraints are that the expected return of the portfolio remains the same.
    """)
    return


@app.cell
def _(cp, np):
    def minimum_variance(
        target_return: float,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        weights = cp.Variable(len(expected_returns))

        objective = cp.Minimize(weights.T @ covariance_matrix @ weights)

        constraints = [
            weights.T @ expected_returns == target_return,
            cp.sum(weights) == 1,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return weights.value
    return (minimum_variance,)


@app.cell
def _(
    covariance_matrix,
    expected_returns,
    minimum_variance,
    np,
    pl,
    weights_list,
):
    min_variance_portfolios_list = []
    for _weights in weights_list:
        target_return = _weights.T @ expected_returns
        min_variance_weights = minimum_variance(
            target_return, expected_returns, covariance_matrix
        )

        _portfolio_return = min_variance_weights.T @ expected_returns * 252
        _portfolio_volatility = np.sqrt(
            min_variance_weights.T @ covariance_matrix @ min_variance_weights
        ) * np.sqrt(252)

        min_variance_portfolios_list.append(
            {
                "ticker": "Minimum Variance",
                "mean_return": _portfolio_return,
                "volatility": _portfolio_volatility,
            }
        )

    min_variance_portfolios = pl.DataFrame(min_variance_portfolios_list)
    return (min_variance_portfolios,)


@app.cell
def _(
    min_variance_portfolios,
    plot_portfolios,
    random_portfolios,
    single_stock_portfolios,
):
    plot_portfolios(
        [single_stock_portfolios, random_portfolios, min_variance_portfolios]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tangent Portfolio
    - Now we will find the optimal portfolio among our universe.
    - We can do this by matrix multiplying the inverse covariance matrix by the excess returns (expected returns minus risk free rate)
    """)
    return


@app.cell
def _(covariance_matrix, expected_returns, np, pl):
    # Assume a risk free rate of 5% annually
    risk_free_rate = .05

    # Compute the excess returns and convert to daily
    excess_returns = expected_returns - risk_free_rate / 252

    # Invert the covariance matrix
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    # Compute the tangent weights
    tangent_weights = inverse_covariance_matrix @ excess_returns

    # Enforce that they sum to one
    tangent_weights = tangent_weights / np.sum(tangent_weights)

    # Calculate tangent portfolio return and volatility
    tangent_return = tangent_weights.T @ expected_returns * 252
    tangent_volatility = np.sqrt(
        tangent_weights.T @ covariance_matrix @ tangent_weights
    ) * np.sqrt(252)

    tangent_portfolio = pl.DataFrame([{
        'ticker': 'Tangent',
        'mean_return': tangent_return,
        'volatility': tangent_volatility,
    }])
    return (
        risk_free_rate,
        tangent_portfolio,
        tangent_return,
        tangent_volatility,
    )


@app.cell
def _(
    min_variance_portfolios,
    plot_portfolios,
    random_portfolios,
    single_stock_portfolios,
    tangent_portfolio,
):
    plot_portfolios(
        [
            single_stock_portfolios,
            random_portfolios,
            min_variance_portfolios,
            tangent_portfolio
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Capital Allocation Line (CAL)
    - The rational investor would invest in the tangent portfolio and tilt towards or away from it by levering in and out of the risk free rate.
    - We can approximate this by creating a line that connects the risk free rate portfolio (y-intercept) to the tangent portfolio.
    """)
    return


@app.cell
def _(np, pl, risk_free_rate, tangent_return, tangent_volatility):
    # Create CAL line passing through risk-free rate and tangent portfolio
    slope = (tangent_return - risk_free_rate) / tangent_volatility
    intercept = risk_free_rate

    bounds = [0, 1.5 * tangent_volatility]
    x = np.linspace(*bounds, 100)
    y = slope * x + intercept

    cal_line_portfolios = pl.DataFrame({
        'ticker': ['CAL'] * 100,
        'mean_return': y,
        'volatility': x,
    })
    return (cal_line_portfolios,)


@app.cell
def _(
    cal_line_portfolios,
    max_utility_portfolio,
    min_variance_portfolios,
    plot_portfolios,
    random_portfolios,
    single_stock_portfolios,
    tangent_portfolio,
):
    plot_portfolios(
        [
            single_stock_portfolios,
            random_portfolios,
            min_variance_portfolios,
            max_utility_portfolio,
            cal_line_portfolios,
            tangent_portfolio
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Maximum Utility Portfolio
    - In practice it can be computationally expensive to invert large covariance matrices, so we use a different objective function in our optimizer: utility.
    - Our utility function is constructed as:
    - Note that our maximum utility portfolio is very close to the capital allocation line.
    """)
    return


@app.cell
def _(cp, np):
    def maximum_utility(
        expected_returns: np.ndarray, covariance_matrix: np.ndarray, lambda_: float = 2
    ) -> np.ndarray:
        weights = cp.Variable(len(expected_returns))

        objective = cp.Maximize(
            weights.T @ expected_returns
            - (lambda_ / 2) * cp.quad_form(weights, covariance_matrix)
        )

        constraints = [
            cp.sum(weights) == 1,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return weights.value
    return (maximum_utility,)


@app.cell
def _(covariance_matrix, expected_returns, maximum_utility, np, pl):
    max_utility_weights = maximum_utility(
        expected_returns, covariance_matrix, lambda_=100
    )

    max_utility_return = max_utility_weights.T @ expected_returns * 252
    max_utility_volatility = np.sqrt(
        max_utility_weights.T @ covariance_matrix @ max_utility_weights
    ) * np.sqrt(252)

    max_utility_portfolio = pl.DataFrame(
        [
            {
                "ticker": "Maximum Utility",
                "mean_return": max_utility_return,
                "volatility": max_utility_volatility,
            }
        ]
    )
    return (max_utility_portfolio,)


@app.cell
def _(
    cal_line_portfolios,
    max_utility_portfolio,
    min_variance_portfolios,
    plot_portfolios,
    random_portfolios,
    single_stock_portfolios,
    tangent_portfolio,
):
    plot_portfolios(
        [
            single_stock_portfolios,
            random_portfolios,
            min_variance_portfolios,
            max_utility_portfolio,
            cal_line_portfolios,
            tangent_portfolio,
            max_utility_portfolio
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
