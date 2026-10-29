import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from textwrap import fill
from io import StringIO
from urllib.request import Request, urlopen

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from pypdf.generic import Fit


# ============================================================
# GLOBAL PDF / MATPLOTLIB SETTINGS
# ============================================================

PAGE_SIZE = (11, 8.5)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 20
})


# ============================================================
# REPORT BRANDING
# ============================================================

REPORT_BANNER = "McGill Office of Investments"
REPORT_TITLE = "Market Sentiment Dashboard"

BANNER_COLOR = "#ED1B2F"

# Conventional hyperlink styling for clickable indicator names on page 1.
HYPERLINK_COLOR = "#0563C1"
BANNER_TEXT_COLOR = "white"


# ============================================================
# TABLE CONDITIONAL COLORS
# ============================================================

CELL_HIGH = "#DDEFE4"
CELL_HIGH_TEXT = "#1F633B"

CELL_LOW = "#F5DDDD"
CELL_LOW_TEXT = "#8C2F2F"

CELL_NEUTRAL = "#F1F2F3"
CELL_NEUTRAL_TEXT = "#333333"

CELL_WATCH = "#F7EECB"
CELL_WATCH_TEXT = "#735C14"

HEADER_BG = "#F0F0F0"


# ============================================================
# SETTINGS
# ============================================================

today = pd.Timestamp.today().normalize()

START_DATE = (
    today
    - pd.DateOffset(years=10, days=30)
).strftime("%Y-%m-%d")

# Trend methodology:
# - Momentum = annualized slope of a 63-trading-day linear regression
#   fitted to the log of each relative-performance ratio.
# - Acceleration = change in that annualized momentum over 21 trading days.
# - Momentum strength is assessed with the regression slope t-statistic.
# - Acceleration strength is assessed with a trailing 3-year z-score.
MOMENTUM_WINDOW = 63
ACCELERATION_LOOKBACK = 21
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# YAHOO FINANCE TICKERS
# ============================================================

YF_TICKERS = {

    # Portfolio Factor Positioning
    "XDG.U": "XDG-U.TO",
    "URTH": "URTH",

    "QUAL": "QUAL",
    "PBUS": "PBUS",

    "IXC": "IXC",
    "MXI": "MXI",

    "IJR": "IJR",
    "IWM": "IWM",

    "EMXC": "EMXC",
    "MCHI": "MCHI",

    "IWF": "IWF",
    "IWD": "IWD",

    # Risk Appetite / Speculation
    "IWC": "IWC",

    "KRE": "KRE",
    "SPY": "SPY",

    "EXV1": "EXV1.DE",
    "E600": "^STOXX",

    "SHYG": "SHYG",
    "IGSB": "IGSB",

    # EM local bond decomposition proxies
    # LEMB: iShares J.P. Morgan EM Local Currency Bond ETF
    # USDCAD: USD/CAD exchange rate
    # ZFM: BMO Mid Federal Bond Index ETF
    #
    # The LEMB-matched currency basket is built dynamically from the
    # latest iShares holdings file, so individual FX tickers do not
    # need to be listed here.
    "LEMB": "LEMB",
    "USDCAD": "CAD=X",
    "ZFS": "ZFS.TO",
    "ZFM": "ZFM.TO"
}


# ============================================================
# OUTPUT FILE
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent

DEFAULT_OUTPUT_PDF = (
    SCRIPT_FOLDER
    / "market_sentiment_report.pdf"
)


def get_available_output_path(default_path):

    try:

        if default_path.exists():

            with open(default_path, "a+b"):
                pass

        return default_path

    except PermissionError:

        timestamp = pd.Timestamp.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        alternative_path = (
            default_path.parent
            / f"market_sentiment_report_{timestamp}.pdf"
        )

        print()
        print("Existing PDF is open or locked.")
        print("Saving new report as:")
        print(alternative_path)

        return alternative_path


OUTPUT_PDF = get_available_output_path(
    DEFAULT_OUTPUT_PDF
)


# ============================================================
# EM LOCAL BOND PROXY FRAMEWORK
# ============================================================
#
# Fully automatic framework using:
#
#   LEMB   = iShares J.P. Morgan EM Local Currency Bond ETF
#   CAD=X  = USD/CAD
#   ZFM.TO = BMO Mid Federal Bond Index ETF
#
# Instead of using CEW as a generic EM-currency proxy, the script:
#
#   1) downloads the latest LEMB holdings CSV from iShares;
#   2) aggregates portfolio weights by each holding's Market Currency;
#   3) downloads the corresponding Yahoo Finance FX series;
#   4) converts each local currency into CAD terms;
#   5) builds a daily-rebalanced currency index using LEMB's currency
#      weights; and
#   6) removes that matched FX index from LEMB's CAD return to estimate
#      the local-rates-and-carry component.
#
# The current currency weights are applied historically. This is still
# a proxy rather than an official hedged J.P. Morgan index, but it is
# materially better matched to LEMB than a separate generic FX ETF.
#
# For the Canadian comparison, the script builds a synthetic federal
# government-bond benchmark from ZFS and ZFM. Fixed weights are chosen
# so its approximate effective duration matches LEMB.
#
# Duration assumptions:
#   LEMB target = 5.20 years
#   ZFS         = 2.87 years
#   ZFM         = 6.57 years
#
# If the live iShares holdings download fails, the script falls back to
# the most recently hard-coded official top-currency weights and reports
# the lower portfolio coverage explicitly.


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

print("Downloading market data...")

yahoo_symbols = list(
    YF_TICKERS.values()
)

raw_prices = yf.download(
    yahoo_symbols,
    start=START_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

raw_prices = raw_prices.dropna(
    how="all"
)


# ============================================================
# RENAME TICKERS
# ============================================================

reverse_mapping = {

    yahoo_ticker: internal_name

    for internal_name, yahoo_ticker
    in YF_TICKERS.items()
}

prices = raw_prices.rename(
    columns=reverse_mapping
)


# ============================================================
# CHECK DATA
# ============================================================

missing_tickers = [

    ticker

    for ticker in YF_TICKERS.keys()

    if (
        ticker not in prices.columns
        or prices[ticker].dropna().empty
    )
]


if missing_tickers:

    print()
    print("WARNING: Missing market data for:")

    for ticker in missing_tickers:
        print(f"  - {ticker}")

    print()


# ============================================================
# BUILD RATIOS
# ============================================================

ratios = pd.DataFrame(
    index=prices.index
)


# ============================================================
# PORTFOLIO FACTOR POSITIONING
# ============================================================


# ------------------------------------------------------------
# 1. HIGH DIVIDEND VS WORLD
# ------------------------------------------------------------

ratios["XDG.U / URTH"] = (
    prices["XDG.U"]
    / prices["URTH"]
)


# ------------------------------------------------------------
# 2. QUALITY VS US MARKET
# ------------------------------------------------------------

ratios["QUAL / PBUS"] = (
    prices["QUAL"]
    / prices["PBUS"]
)


# ------------------------------------------------------------
# 3. RESOURCES VS WORLD
#
# 50/50 daily-rebalanced IXC + MXI basket
# ------------------------------------------------------------

resource_prices = pd.concat(
    [
        prices["IXC"],
        prices["MXI"]
    ],
    axis=1
).dropna()

resource_prices.columns = [
    "IXC",
    "MXI"
]

resource_returns = (
    resource_prices
    .pct_change()
)

resource_equal_weight_return = (

    0.50 * resource_returns["IXC"]

    +

    0.50 * resource_returns["MXI"]
)

resource_index = (
    1
    + resource_equal_weight_return.fillna(0)
).cumprod()

urth_resources = (
    prices["URTH"]
    .reindex(resource_index.index)
    .dropna()
)

common_resource_dates = (
    resource_index.index.intersection(
        urth_resources.index
    )
)

resource_index = (
    resource_index.loc[
        common_resource_dates
    ]
)

urth_resources = (
    urth_resources.loc[
        common_resource_dates
    ]
)

urth_normalized = (
    urth_resources
    / urth_resources.iloc[0]
)

resources_vs_world = (
    resource_index
    / urth_normalized
)

ratios.loc[
    resources_vs_world.index,
    "50/50 IXC + MXI / URTH"
] = resources_vs_world


# ------------------------------------------------------------
# 4. PROFITABLE SMALL CAPS VS BROAD SMALL CAPS
# ------------------------------------------------------------

ratios["IJR / IWM"] = (
    prices["IJR"]
    / prices["IWM"]
)


# ------------------------------------------------------------
# 5. EM EX CHINA VS CHINA
# ------------------------------------------------------------

ratios["EMXC / MCHI"] = (
    prices["EMXC"]
    / prices["MCHI"]
)


# ------------------------------------------------------------
# 6. GROWTH VS VALUE
# ------------------------------------------------------------

ratios["IWF / IWD"] = (
    prices["IWF"]
    / prices["IWD"]
)


# ============================================================
# RISK APPETITE / SPECULATION
# ============================================================


# ------------------------------------------------------------
# 7. MICROCAPS VS PROFITABLE SMALL CAPS
# ------------------------------------------------------------

ratios["IWC / IJR"] = (
    prices["IWC"]
    / prices["IJR"]
)


# ------------------------------------------------------------
# 8. US REGIONAL BANKS
# ------------------------------------------------------------

ratios["KRE / SPY"] = (
    prices["KRE"]
    / prices["SPY"]
)


# ------------------------------------------------------------
# 9. EUROPEAN BANKS
# ------------------------------------------------------------

ratios["EXV1 / E600"] = (
    prices["EXV1"]
    / prices["E600"]
)


# ------------------------------------------------------------
# 10. HIGH YIELD VS INVESTMENT GRADE
# ------------------------------------------------------------

ratios["SHYG / IGSB"] = (
    prices["SHYG"]
    / prices["IGSB"]
)


# ============================================================
# INDICATOR GROUPS
# ============================================================

PORTFOLIO_FACTORS = [

    "XDG.U / URTH",

    "QUAL / PBUS",

    "50/50 IXC + MXI / URTH",

    "IJR / IWM",

    "EMXC / MCHI",

    "IWF / IWD"
]


RISK_APPETITE = [

    "IWC / IJR",

    "KRE / SPY",

    "EXV1 / E600",

    "SHYG / IGSB"
]


EM_LOCAL_SIGNALS = [

    "EM Local / Synthetic Canada",

    "LEMB-Matched EM FX / CAD"
]


# ============================================================
# SHORT NAMES
# ============================================================

short_names = {

    "XDG.U / URTH":
        "High Dividend vs World",

    "QUAL / PBUS":
        "Quality vs US Market",

    "50/50 IXC + MXI / URTH":
        "Resources vs World",

    "IJR / IWM":
        "Profitable vs Broad Small Caps",

    "EMXC / MCHI":
        "EM ex China vs China",

    "IWF / IWD":
        "Growth vs Value",

    "IWC / IJR":
        "Small-Cap Speculation",

    "KRE / SPY":
        "US Regional Banks",

    "EXV1 / E600":
        "European Banks",

    "SHYG / IGSB":
        "High Yield vs Investment Grade",

    "EM Local / Synthetic Canada":
        "EM Local Bond Index vs Canada Federal Bond Index",

    "LEMB-Matched EM FX / CAD":
        "LEMB-Matched EM FX vs CAD"
}


# ============================================================
# PURPOSES
# ============================================================

indicator_purposes = {

    "XDG.U / URTH":
        "Compares the MSCI World High Dividend Yield factor "
        "with the broad MSCI World market.",

    "QUAL / PBUS":
        "Measures quality-factor leadership within a consistent "
        "MSCI USA equity universe.",

    "50/50 IXC + MXI / URTH":
        "Measures an equal-weighted global Energy and Materials "
        "basket relative to broad global equities.",

    "IJR / IWM":
        "Measures the relative leadership of profitable, "
        "higher-quality small-cap companies versus the broader "
        "small-cap universe.",

    "EMXC / MCHI":
        "Measures emerging-market ex-China leadership relative "
        "to China. A rising ratio indicates EM ex-China "
        "outperformance; a falling ratio indicates China leadership.",

    "IWF / IWD":
        "Measures US large-cap growth leadership relative to "
        "US large-cap value within the Russell 1000 universe.",

    "IWC / IJR":
        "Measures microcap leadership relative to profitable "
        "small caps. Higher readings indicate stronger "
        "speculative risk appetite.",

    "KRE / SPY":
        "Measures US regional-bank leadership relative to the "
        "broad US equity market and provides a domestic banking "
        "and credit-cycle signal.",

    "EXV1 / E600":
        "Measures European bank leadership relative to the "
        "STOXX Europe 600 and provides a European banking and "
        "economic-cycle signal.",

    "SHYG / IGSB":
        "Compares short-duration high-yield credit with "
        "short-duration investment-grade credit to better "
        "isolate corporate credit risk appetite.",

    "EM Local / Synthetic Canada":
        "Compares a LEMB-derived EM local government bond index "
        "(local rates & carry, with FX stripped out) against a "
        "duration-matched synthetic Canadian federal government bond "
        "index. A rising ratio indicates EM local sovereign bonds are "
        "outperforming comparable-duration Canadian federal bonds.",

    "LEMB-Matched EM FX / CAD":
        "Measures the CAD value of a currency basket matched to LEMB's "
        "latest market-currency weights. A rising index indicates the "
        "LEMB-weighted EM currency basket is strengthening versus CAD."
}


# ============================================================
# BUCKETS
# ============================================================

indicator_buckets = {

    ratio_name: (
        "Portfolio Factor Positioning"
        if ratio_name in PORTFOLIO_FACTORS
        else "Risk Appetite / Speculation"
        if ratio_name in RISK_APPETITE
        else "EM Local Bond / Currency"
    )

    for ratio_name in (
        PORTFOLIO_FACTORS
        + RISK_APPETITE
        + EM_LOCAL_SIGNALS
    )
}


# ============================================================
# CALCULATION FUNCTIONS
# ============================================================

def trailing_data(
    series,
    years
):

    start_date = (
        today
        - pd.DateOffset(years=years)
    )

    return series.loc[
        (series.index >= start_date)
        & (series.index <= today)
    ].dropna()


def trailing_average(
    series,
    years
):

    data = trailing_data(
        series,
        years
    )

    if data.empty:
        return float("nan")

    return data.mean()


def latest_ratio(series):

    data = series.dropna()

    if data.empty:
        return float("nan")

    return data.iloc[-1]


def latest_date(series):

    data = series.dropna()

    if data.empty:
        return None

    return data.index[-1]


def percentage_deviation(
    current_value,
    average_value
):

    if (
        pd.isna(current_value)
        or pd.isna(average_value)
        or average_value == 0
    ):

        return float("nan")

    return (
        current_value
        / average_value
        - 1
    ) * 100


def percentage_change(
    current_value,
    previous_value
):

    if (
        pd.isna(current_value)
        or pd.isna(previous_value)
        or previous_value == 0
    ):

        return float("nan")

    return (
        current_value
        / previous_value
        - 1
    ) * 100


def z_score(
    series,
    current_value,
    years=3
):

    data = trailing_data(
        series,
        years
    )

    if (
        data.empty
        or pd.isna(current_value)
    ):

        return float("nan")

    mean = data.mean()
    std = data.std()

    if (
        pd.isna(std)
        or std == 0
    ):

        return float("nan")

    return (
        current_value
        - mean
    ) / std


def historical_percentile(
    series,
    current_value,
    years=3
):

    data = trailing_data(
        series,
        years
    )

    if (
        data.empty
        or pd.isna(current_value)
    ):

        return float("nan")

    return (
        data <= current_value
    ).mean() * 100


def available_years(series):

    data = series.dropna()

    if len(data) < 2:
        return 0.0

    days = (
        data.index[-1]
        - data.index[0]
    ).days

    return (
        days
        / 365.25
    )


def rolling_log_trend(
    series,
    window=MOMENTUM_WINDOW
):
    """
    Estimate a rolling linear trend in log(relative ratio).

    Returns:
        annualized_slope_pct:
            Approximate annualized relative-performance trend in percent.
        slope_tstat:
            OLS t-statistic of the slope inside each rolling window.
    """

    clean = series.dropna()

    log_series = np.log(
        clean
    )

    annualized_slope_pct = pd.Series(
        index=log_series.index,
        dtype=float
    )

    slope_tstat = pd.Series(
        index=log_series.index,
        dtype=float
    )

    if len(log_series) < window:
        return (
            annualized_slope_pct,
            slope_tstat
        )

    x = np.arange(
        window,
        dtype=float
    )

    x_mean = x.mean()

    x_centered = (
        x
        - x_mean
    )

    sxx = np.sum(
        x_centered ** 2
    )

    for i in range(
        window - 1,
        len(log_series)
    ):

        y = log_series.iloc[
            i - window + 1:
            i + 1
        ].values.astype(float)

        y_mean = y.mean()

        slope = (
            np.sum(
                x_centered
                * (y - y_mean)
            )
            / sxx
        )

        intercept = (
            y_mean
            - slope * x_mean
        )

        fitted = (
            intercept
            + slope * x
        )

        residuals = (
            y
            - fitted
        )

        dof = (
            window
            - 2
        )

        if dof > 0:

            residual_variance = (
                np.sum(
                    residuals ** 2
                )
                / dof
            )

            slope_se = np.sqrt(
                residual_variance
                / sxx
            )

        else:

            slope_se = float("nan")

        annualized_slope_pct.iloc[i] = (
            slope
            * TRADING_DAYS_PER_YEAR
            * 100
        )

        if (
            not pd.isna(slope_se)
            and slope_se > 0
        ):

            slope_tstat.iloc[i] = (
                slope
                / slope_se
            )

    return (
        annualized_slope_pct,
        slope_tstat
    )


def momentum_acceleration_series(
    annualized_momentum,
    lookback=ACCELERATION_LOOKBACK
):
    """
    Finite-difference approximation to the second derivative.

    Acceleration is the change in annualized momentum over the
    previous lookback trading days, expressed in annualized
    percentage-point terms.
    """

    return (
        annualized_momentum
        - annualized_momentum.shift(
            lookback
        )
    )


def trailing_zscore_of_series(
    series,
    years=3
):

    clean = series.dropna()

    if clean.empty:
        return float("nan")

    current_value = clean.iloc[-1]

    trailing = trailing_data(
        clean,
        years
    )

    if trailing.empty:
        return float("nan")

    mean = trailing.mean()
    std = trailing.std()

    if (
        pd.isna(std)
        or std == 0
    ):

        return float("nan")

    return (
        current_value
        - mean
    ) / std



def fitted_log_trend_line(
    series,
    window=MOMENTUM_WINDOW,
    end_offset=0
):
    """
    Return a straight visual trend line in ratio space while preserving
    the exact same log-linear regression slope used for momentum.

    The momentum calculation itself remains:
        log(Relative Ratio) = alpha + beta * time

    For chart presentation only, the fitted regression endpoints are
    transformed back into ratio space and connected with a straight line.

    end_offset=0:
        current 63-trading-day trend

    end_offset=ACCELERATION_LOOKBACK:
        63-day trend as of 21 trading days ago

    Returns:
        dates
        straight_visual_ratio_line
        annualized_slope_pct
    """

    clean = series.dropna()

    end_position = (
        len(clean)
        - 1
        - end_offset
    )

    start_position = (
        end_position
        - window
        + 1
    )

    if (
        end_position < 0
        or start_position < 0
    ):
        return (
            pd.DatetimeIndex([]),
            np.array([]),
            float("nan")
        )

    window_series = clean.iloc[
        start_position:
        end_position + 1
    ]

    y = np.log(
        window_series.values.astype(float)
    )

    x = np.arange(
        len(window_series),
        dtype=float
    )

    x_mean = x.mean()

    x_centered = (
        x
        - x_mean
    )

    sxx = np.sum(
        x_centered ** 2
    )

    if sxx == 0:

        return (
            window_series.index,
            np.full(
                len(window_series),
                np.nan
            ),
            float("nan")
        )

    y_mean = y.mean()

    slope = (
        np.sum(
            x_centered
            * (y - y_mean)
        )
        / sxx
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    fitted_log_start = (
        intercept
        + slope * x[0]
    )

    fitted_log_end = (
        intercept
        + slope * x[-1]
    )

    fitted_ratio_start = np.exp(
        fitted_log_start
    )

    fitted_ratio_end = np.exp(
        fitted_log_end
    )

    # Straight line for visual presentation.
    straight_visual_ratio_line = np.linspace(
        fitted_ratio_start,
        fitted_ratio_end,
        len(window_series)
    )

    annualized_slope_pct = (
        slope
        * TRADING_DAYS_PER_YEAR
        * 100
    )

    return (
        window_series.index,
        straight_visual_ratio_line,
        annualized_slope_pct
    )


# ============================================================
# POSITIONING / MOMENTUM / ACCELERATION LABELS
# ============================================================

def positioning_label(z):

    if pd.isna(z):
        return "No Signal"

    if z >= 2:
        return "Extremely High"

    elif z >= 1:
        return "High"

    elif z > -1:
        return "Neutral"

    elif z > -2:
        return "Low"

    else:
        return "Extremely Low"


def momentum_label(
    momentum_value,
    momentum_tstat
):

    if (
        pd.isna(momentum_value)
        or pd.isna(momentum_tstat)
    ):
        return "Unavailable"

    if (
        momentum_value > 0
        and momentum_tstat >= 2
    ):
        return "Strong Positive"

    if (
        momentum_value > 0
        and momentum_tstat >= 1
    ):
        return "Positive"

    if (
        momentum_value < 0
        and momentum_tstat <= -2
    ):
        return "Strong Negative"

    if (
        momentum_value < 0
        and momentum_tstat <= -1
    ):
        return "Negative"

    return "Flat / Noisy"


def acceleration_label(
    acceleration_value,
    acceleration_z
):

    if (
        pd.isna(acceleration_value)
        or pd.isna(acceleration_z)
    ):
        return "Unavailable"

    if (
        acceleration_value > 0
        and acceleration_z >= 1
    ):
        return "Strongly Accelerating"

    if acceleration_value > 0:
        return "Accelerating"

    if (
        acceleration_value < 0
        and acceleration_z <= -1
    ):
        return "Strongly Decelerating"

    if acceleration_value < 0:
        return "Decelerating"

    return "Stable"


# ============================================================
# FORMAT FUNCTIONS
# ============================================================

def format_number(
    value,
    decimals=4
):

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def format_signed(
    value,
    decimals=2
):

    if pd.isna(value):
        return "N/A"

    return f"{value:+.{decimals}f}"


def format_percent(
    value,
    decimals=1
):

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


def format_signed_percent(
    value,
    decimals=2
):

    if pd.isna(value):
        return "N/A"

    return f"{value:+.{decimals}f}%"



# ============================================================
# EM LOCAL BOND DECOMPOSITION - LEMB-MATCHED FX BASKET
# ============================================================

LEMB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239528/"
    "ishares-j-p-morgan-em-local-currency-bond-etf/"
    "latest-holdings.csv"
)


# Official currency weights displayed by iShares on Sep 23, 2026.
# These are used only as a fallback if the live holdings CSV cannot
# be downloaded. Because only the publicly displayed top currencies
# are included here, the fallback intentionally reports partial
# coverage rather than pretending to be a full portfolio replication.
LEMB_FALLBACK_CURRENCY_WEIGHTS = {
    "CNY": 15.25,
    "INR": 8.91,
    "MXN": 5.49,
    "MYR": 4.87,
    "IDR": 4.86,
    "PLN": 4.43,
    "THB": 4.22,
    "BRL": 4.11,
    "ZAR": 4.09,
    "UYU": 4.08,
    "TRY": 4.07,
    "PYG": 4.05
}


def normalize_to_100(
    series
):
    """
    Normalize a positive index-like series to 100 at the first
    available observation.
    """

    clean = (
        series
        .dropna()
        .sort_index()
    )

    if clean.empty:
        return clean

    return (
        clean
        / clean.iloc[0]
        * 100
    )


def total_return_between(
    series,
    start_date,
    end_date=None
):
    """
    Total return in percent from the first available observation
    on/after start_date through the last observation on/before end_date.
    """

    clean = (
        series
        .dropna()
        .sort_index()
    )

    if clean.empty:
        return float("nan")

    start_date = pd.Timestamp(
        start_date
    )

    period = clean.loc[
        clean.index >= start_date
    ]

    if end_date is not None:

        end_date = pd.Timestamp(
            end_date
        )

        period = period.loc[
            period.index <= end_date
        ]

    if len(period) < 2:
        return float("nan")

    return (
        period.iloc[-1]
        / period.iloc[0]
        - 1
    ) * 100


def rolling_total_return(
    series,
    periods=252
):
    """
    Rolling total return in percent using a trading-day lookback.
    """

    clean = (
        series
        .dropna()
        .sort_index()
    )

    return (
        clean
        / clean.shift(
            periods
        )
        - 1
    ) * 100


def rolling_zscore_series(
    series,
    periods=756,
    min_periods=252
):
    """
    Rolling z-score of a signal.

    Default window:
        756 trading days ~= 3 years

    A minimum of 252 observations is required so the chart can begin
    before a full three years of history have accumulated.
    """

    clean = (
        series
        .dropna()
        .sort_index()
    )

    rolling_mean = (
        clean
        .rolling(
            window=periods,
            min_periods=min_periods
        )
        .mean()
    )

    rolling_std = (
        clean
        .rolling(
            window=periods,
            min_periods=min_periods
        )
        .std()
    )

    z = (
        clean
        - rolling_mean
    ) / rolling_std

    return (
        z
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


def download_lemb_currency_weights():
    """
    Download the latest iShares LEMB holdings CSV and aggregate
    holding weights by Market Currency.

    The holdings file contains a 'Market Currency' field, which is the
    economically relevant local currency exposure for each bond.

    Returns
    -------
    weights : pandas Series
        Currency weights in percentage points, e.g. CNY = 15.2.
    holdings_as_of : str
        iShares holdings date when available.
    source_label : str
        'Live iShares holdings' or 'Fallback official top weights'.
    """

    request = Request(
        LEMB_HOLDINGS_URL,
        headers={
            "User-Agent":
                (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/140 Safari/537.36"
                )
        }
    )

    try:

        with urlopen(
            request,
            timeout=20
        ) as response:

            raw_text = (
                response
                .read()
                .decode(
                    "utf-8-sig",
                    errors="replace"
                )
            )


        lines = raw_text.splitlines()

        holdings_as_of = "Latest"

        for line in lines[:10]:

            if line.startswith(
                "Fund Holdings as of"
            ):

                parts = line.split(
                    ",",
                    1
                )

                if len(parts) == 2:

                    holdings_as_of = (
                        parts[1]
                        .strip()
                        .strip('"')
                    )

                break


        header_index = None

        for i, line in enumerate(
            lines
        ):

            if (
                line.startswith(
                    "Name,Sector,Asset Class"
                )
                and "Market Currency" in line
            ):

                header_index = i
                break


        if header_index is None:

            raise ValueError(
                "Could not locate holdings CSV header."
            )


        holdings = pd.read_csv(
            StringIO(
                "\n".join(
                    lines[
                        header_index:
                    ]
                )
            )
        )


        required_columns = [
            "Weight (%)",
            "Market Currency"
        ]


        for column in required_columns:

            if column not in holdings.columns:

                raise ValueError(
                    f"Missing expected holdings column: {column}"
                )


        holdings[
            "Weight (%)"
        ] = pd.to_numeric(
            holdings[
                "Weight (%)"
            ],
            errors="coerce"
        )


        holdings[
            "Market Currency"
        ] = (
            holdings[
                "Market Currency"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )


        usable = holdings.loc[
            (
                holdings[
                    "Weight (%)"
                ].notna()
            )
            &
            (
                holdings[
                    "Market Currency"
                ].notna()
            )
            &
            (
                holdings[
                    "Market Currency"
                ].ne("-")
            )
            &
            (
                holdings[
                    "Market Currency"
                ].ne("NAN")
            )
        ].copy()


        weights = (
            usable
            .groupby(
                "Market Currency"
            )[
                "Weight (%)"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )


        # Remove any zero/negative operational rows.
        weights = weights.loc[
            weights > 0
        ]


        if weights.empty:

            raise ValueError(
                "No usable currency weights found."
            )


        print()
        print(
            "Loaded live LEMB currency weights from iShares "
            f"({holdings_as_of})."
        )

        print(
            f"Holdings currency weights sum to "
            f"{weights.sum():.2f}%."
        )


        return (
            weights,
            holdings_as_of,
            "Live iShares holdings"
        )


    except Exception as exc:

        print()
        print(
            "WARNING: Could not download/parse live LEMB holdings."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Using official fallback top-currency weights instead."
        )


        weights = pd.Series(
            LEMB_FALLBACK_CURRENCY_WEIGHTS,
            dtype=float
        ).sort_values(
            ascending=False
        )


        return (
            weights,
            "2026-09-23",
            "Fallback official top weights"
        )


def download_currency_fx_data(
    currency_codes,
    start_date
):
    """
    Download Yahoo Finance FX rates needed to convert each LEMB
    market currency into CAD.

    Yahoo convention used here:
        CAD=X  -> CAD per USD
        MXN=X  -> MXN per USD
        INR=X  -> INR per USD
        etc.

    Therefore:
        CAD per Local Currency
            = (CAD per USD) / (Local Currency per USD)
            = USD/CAD / USD/Local in market quoting language.
    """

    local_codes = sorted(
        {
            str(code).upper()

            for code in currency_codes

            if str(code).upper()
            not in {
                "CAD",
                "USD"
            }
        }
    )


    yahoo_tickers = [
        f"{code}=X"
        for code in local_codes
    ]


    if not yahoo_tickers:

        return pd.DataFrame()


    print()
    print(
        "Downloading LEMB-matched currency FX data..."
    )


    raw_fx = yf.download(
        yahoo_tickers,
        start=start_date,
        auto_adjust=False,
        progress=False
    )


    if raw_fx.empty:
        return pd.DataFrame()


    if isinstance(
        raw_fx.columns,
        pd.MultiIndex
    ):

        if "Close" in raw_fx.columns.get_level_values(0):

            fx_close = raw_fx[
                "Close"
            ].copy()

        else:

            return pd.DataFrame()

    else:

        # yfinance may return a flat frame for one ticker.
        if "Close" in raw_fx.columns:

            fx_close = raw_fx[
                ["Close"]
            ].copy()

            if len(
                yahoo_tickers
            ) == 1:

                fx_close.columns = [
                    yahoo_tickers[0]
                ]

        else:

            return pd.DataFrame()


    if isinstance(
        fx_close,
        pd.Series
    ):

        fx_close = (
            fx_close
            .to_frame(
                name=yahoo_tickers[0]
            )
        )


    return (
        fx_close
        .sort_index()
    )


def build_lemb_matched_fx_index(
    prices,
    currency_weights
):
    """
    Build a current-weight LEMB-matched currency basket in CAD.

    Each local currency is converted into CAD terms using:

        CAD per local currency
            = (CAD per USD) / (local currency per USD)

    Daily currency returns are then combined using the latest LEMB
    currency weights. The weights are applied historically and the
    basket is daily rebalanced.

    Currencies with insufficient Yahoo Finance history are omitted and
    the remaining covered weights are re-normalized to 100%. Portfolio
    weight coverage is reported so the approximation remains transparent.

    Returns
    -------
    fx_index : pandas Series
        LEMB-matched FX basket indexed to 100.
    used_weights : pandas Series
        Original LEMB weights for currencies successfully represented.
    coverage_pct : float
        Sum of original LEMB weights represented in the basket.
    missing_currencies : list
        Currency codes omitted because Yahoo data were unavailable.
    """

    if (
        "LEMB" not in prices.columns
        or prices[
            "LEMB"
        ].dropna().empty
        or "USDCAD" not in prices.columns
        or prices[
            "USDCAD"
        ].dropna().empty
    ):

        return (
            pd.Series(
                dtype=float
            ),
            pd.Series(
                dtype=float
            ),
            0.0,
            list(
                currency_weights.index
            )
        )


    lemb_dates = (
        prices[
            "LEMB"
        ]
        .dropna()
        .index
    )


    usdcad = (
        prices[
            "USDCAD"
        ]
        .reindex(
            lemb_dates
        )
        .ffill()
        .bfill()
    )


    fx_data = download_currency_fx_data(
        currency_weights.index,
        START_DATE
    )


    cad_value_series = {}
    used_weights = {}
    missing_currencies = []


    for currency, weight in (
        currency_weights.items()
    ):

        currency = str(
            currency
        ).upper()


        # ----------------------------------------------------
        # CAD
        # ----------------------------------------------------

        if currency == "CAD":

            cad_per_currency = pd.Series(
                1.0,
                index=lemb_dates,
                dtype=float
            )


        # ----------------------------------------------------
        # USD
        # ----------------------------------------------------

        elif currency == "USD":

            cad_per_currency = (
                usdcad.copy()
            )


        # ----------------------------------------------------
        # OTHER LOCAL CURRENCIES
        # ----------------------------------------------------

        else:

            ticker = (
                f"{currency}=X"
            )


            if (
                ticker not in fx_data.columns
                or fx_data[
                    ticker
                ].dropna().empty
            ):

                missing_currencies.append(
                    currency
                )

                continue


            local_per_usd = (
                fx_data[
                    ticker
                ]
                .reindex(
                    lemb_dates
                )
                .ffill()
            )


            # Require usable data near the beginning of the LEMB
            # analysis window so a late-starting series does not
            # silently shorten the whole matched basket.
            valid_ratio = (
                local_per_usd
                .notna()
                .mean()
            )


            if valid_ratio < 0.90:

                missing_currencies.append(
                    currency
                )

                continue


            local_per_usd = (
                local_per_usd
                .bfill()
            )


            cad_per_currency = (
                usdcad
                / local_per_usd
            )


        cad_per_currency = (
            cad_per_currency
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .ffill()
            .bfill()
        )


        if cad_per_currency.dropna().empty:

            missing_currencies.append(
                currency
            )

            continue


        cad_value_series[
            currency
        ] = cad_per_currency


        used_weights[
            currency
        ] = float(
            weight
        )


    if not cad_value_series:

        return (
            pd.Series(
                dtype=float
            ),
            pd.Series(
                dtype=float
            ),
            0.0,
            missing_currencies
        )


    used_weights = pd.Series(
        used_weights,
        dtype=float
    ).sort_values(
        ascending=False
    )


    coverage_pct = float(
        used_weights.sum()
    )


    normalized_weights = (
        used_weights
        / used_weights.sum()
    )


    cad_value_frame = pd.DataFrame(
        cad_value_series
    ).reindex(
        lemb_dates
    )


    currency_returns = (
        cad_value_frame
        .pct_change(
            fill_method=None
        )
        .fillna(0.0)
    )


    basket_returns = pd.Series(
        0.0,
        index=currency_returns.index,
        dtype=float
    )


    for currency, normalized_weight in (
        normalized_weights.items()
    ):

        basket_returns = (
            basket_returns
            + currency_returns[
                currency
            ]
            * normalized_weight
        )


    fx_index = (
        (
            1
            + basket_returns
        )
        .cumprod()
        * 100
    )


    fx_index.name = (
        "LEMB-Matched FX Basket"
    )


    print()
    print(
        f"LEMB-matched FX basket coverage: "
        f"{coverage_pct:.2f}% of reported portfolio weight."
    )


    if missing_currencies:

        print(
            "Currencies omitted for insufficient Yahoo data: "
            + ", ".join(
                missing_currencies
            )
        )


    return (
        fx_index,
        used_weights,
        coverage_pct,
        missing_currencies
    )



# ============================================================
# SYNTHETIC CANADIAN FEDERAL BOND BENCHMARK
# ============================================================

LEMB_TARGET_DURATION = 5.20
ZFS_EFFECTIVE_DURATION = 2.87
ZFM_EFFECTIVE_DURATION = 6.57


def duration_match_weights(
    target_duration,
    short_duration,
    long_duration
):
    """
    Solve weights such that:
        w_short * short_duration + w_long * long_duration
        = target_duration
    and:
        w_short + w_long = 1
    """

    denominator = (
        long_duration
        - short_duration
    )

    if denominator == 0:
        raise ValueError(
            "Short and long durations cannot be equal."
        )

    long_weight = (
        target_duration
        - short_duration
    ) / denominator

    short_weight = (
        1.0
        - long_weight
    )

    if not (
        0 <= short_weight <= 1
        and 0 <= long_weight <= 1
    ):
        raise ValueError(
            "Target duration must lie between component durations."
        )

    return (
        short_weight,
        long_weight
    )


(
    CANADA_ZFS_WEIGHT,
    CANADA_ZFM_WEIGHT
) = duration_match_weights(
    LEMB_TARGET_DURATION,
    ZFS_EFFECTIVE_DURATION,
    ZFM_EFFECTIVE_DURATION
)


SYNTHETIC_CANADA_DURATION = (
    CANADA_ZFS_WEIGHT
    * ZFS_EFFECTIVE_DURATION
    +
    CANADA_ZFM_WEIGHT
    * ZFM_EFFECTIVE_DURATION
)


def build_synthetic_canadian_federal_index(
    prices
):
    """
    Daily-rebalanced synthetic Canadian federal-bond benchmark.

    Components:
        ZFS = BMO Short Federal Bond Index ETF
        ZFM = BMO Mid Federal Bond Index ETF

    Fixed weights are selected to target LEMB's 5.20-year duration.
    yfinance prices are downloaded with auto_adjust=True, so the
    adjusted-price returns incorporate distributions.
    """

    for ticker in [
        "ZFS",
        "ZFM"
    ]:

        if (
            ticker not in prices.columns
            or prices[
                ticker
            ].dropna().empty
        ):

            return pd.Series(
                dtype=float,
                name="Synthetic Canada Federal Bonds"
            )


    components = pd.concat(
        [
            prices[
                "ZFS"
            ].rename(
                "ZFS"
            ),

            prices[
                "ZFM"
            ].rename(
                "ZFM"
            )
        ],
        axis=1,
        join="inner"
    ).dropna()


    if len(
        components
    ) < 2:

        return pd.Series(
            dtype=float,
            name="Synthetic Canada Federal Bonds"
        )


    component_returns = (
        components
        .pct_change(
            fill_method=None
        )
        .fillna(0.0)
    )


    synthetic_returns = (
        CANADA_ZFS_WEIGHT
        * component_returns[
            "ZFS"
        ]
        +
        CANADA_ZFM_WEIGHT
        * component_returns[
            "ZFM"
        ]
    )


    synthetic_index = (
        (
            1
            + synthetic_returns
        )
        .cumprod()
        * 100
    )


    synthetic_index.name = (
        "Synthetic Canada Federal Bonds"
    )


    return synthetic_index


def build_em_local_bond_proxy_decomposition(
    prices
):
    """
    Decompose LEMB's CAD return using a currency basket matched to
    LEMB's latest reported market-currency weights.

    CONSTRUCTION
    ------------
    EM Unhedged CAD Proxy
        LEMB adjusted price * USD/CAD

    Matched FX Basket
        Daily-rebalanced weighted index of each LEMB market currency
        versus CAD, using current LEMB currency weights.

    Local Rates & Carry Proxy
        EM Unhedged CAD / Matched FX Basket

    Because:
        Total Index ~= Local Component * FX Component

    the period-return decomposition is:

        Total Return
            = Local Return
            + FX Return
            + Local/FX Interaction

    LIMITATION
    ----------
    Current LEMB currency weights are applied historically. This is much
    better matched than CEW, but it is still not the same as having
    historical daily benchmark weights or an official CAD-hedged index.
    """

    required = [
        "LEMB",
        "USDCAD",
        "ZFS",
        "ZFM"
    ]


    missing = []

    for ticker in required:

        if (
            ticker not in prices.columns
            or prices[
                ticker
            ].dropna().empty
        ):

            missing.append(
                ticker
            )


    if missing:

        print()
        print(
            "EM bond decomposition unavailable. Missing data for: "
            + ", ".join(
                missing
            )
        )

        return None


    (
        currency_weights,
        holdings_as_of,
        weight_source
    ) = download_lemb_currency_weights()


    (
        matched_fx_index,
        used_currency_weights,
        fx_weight_coverage,
        missing_fx_currencies
    ) = build_lemb_matched_fx_index(
        prices,
        currency_weights
    )


    if matched_fx_index.empty:

        print()
        print(
            "Could not construct the LEMB-matched FX basket."
        )

        return None


    synthetic_canada = (
        build_synthetic_canadian_federal_index(
            prices
        )
    )


    if synthetic_canada.empty:

        print()
        print(
            "Could not construct the synthetic Canadian federal-bond benchmark."
        )

        return None


    print()
    print(
        "Synthetic Canada federal benchmark: "
        f"{CANADA_ZFS_WEIGHT:.1%} ZFS + "
        f"{CANADA_ZFM_WEIGHT:.1%} ZFM"
    )

    print(
        "Synthetic Canada duration: "
        f"{SYNTHETIC_CANADA_DURATION:.2f} years "
        f"(target {LEMB_TARGET_DURATION:.2f})"
    )


    base_data = pd.concat(
        [
            prices[
                "LEMB"
            ].rename(
                "LEMB_USD"
            ),

            prices[
                "USDCAD"
            ].rename(
                "USDCAD"
            ),

            synthetic_canada.rename(
                "CANADA_SYNTHETIC_FEDERAL"
            ),

            matched_fx_index.rename(
                "MATCHED_FX"
            )
        ],
        axis=1,
        join="inner"
    ).dropna()


    if len(
        base_data
    ) < 2:

        print()
        print(
            "Insufficient overlapping data for EM bond decomposition."
        )

        return None


    # --------------------------------------------------------
    # BUILD CAD TOTAL-RETURN-LIKE LEMB SERIES
    # --------------------------------------------------------

    em_unhedged_cad_raw = (
        base_data[
            "LEMB_USD"
        ]
        * base_data[
            "USDCAD"
        ]
    )


    em_unhedged_cad = (
        normalize_to_100(
            em_unhedged_cad_raw
        )
        .rename(
            "EM Unhedged CAD Proxy"
        )
    )


    fx_proxy_cad = (
        normalize_to_100(
            base_data[
                "MATCHED_FX"
            ]
        )
        .rename(
            "LEMB-Matched FX Basket"
        )
    )


    # --------------------------------------------------------
    # REMOVE THE MATCHED CURRENCY INDEX
    # --------------------------------------------------------

    local_rates_carry_raw = (
        em_unhedged_cad
        / fx_proxy_cad
    )


    local_rates_carry = (
        normalize_to_100(
            local_rates_carry_raw
        )
        .rename(
            "Local Rates & Carry Proxy"
        )
    )


    canada_bonds = (
        normalize_to_100(
            base_data[
                "CANADA_SYNTHETIC_FEDERAL"
            ]
        )
        .rename(
            "Synthetic Canada Federal Bonds"
        )
    )


    combined = pd.concat(
        [
            em_unhedged_cad,
            fx_proxy_cad,
            local_rates_carry,
            canada_bonds
        ],
        axis=1,
        join="inner"
    ).dropna()


    if len(
        combined
    ) < 2:

        return None


    em_unhedged_cad = (
        combined[
            "EM Unhedged CAD Proxy"
        ]
    )


    fx_proxy_cad = (
        combined[
            "LEMB-Matched FX Basket"
        ]
    )


    local_rates_carry = (
        combined[
            "Local Rates & Carry Proxy"
        ]
    )


    canada_bonds = (
        combined[
            "Synthetic Canada Federal Bonds"
        ]
    )


    # --------------------------------------------------------
    # RELATIVE INDEXES
    # --------------------------------------------------------

    local_vs_canada_index = (
        normalize_to_100(
            local_rates_carry
            / canada_bonds
        )
        .rename(
            "EM Local / Synthetic Canada"
        )
    )


    fx_impact_index = (
        normalize_to_100(
            fx_proxy_cad
        )
        .rename(
            "Currency Impact"
        )
    )


    # --------------------------------------------------------
    # PERIOD ATTRIBUTION
    # --------------------------------------------------------

    latest_date = (
        combined.index[-1]
    )


    ytd_start = pd.Timestamp(
        year=latest_date.year,
        month=1,
        day=1
    )


    one_year_start = (
        latest_date
        - pd.DateOffset(
            years=1
        )
    )


    three_year_start = (
        latest_date
        - pd.DateOffset(
            years=3
        )
    )


    def calculate_period(
        start_date
    ):

        total_return = (
            total_return_between(
                em_unhedged_cad,
                start_date,
                latest_date
            )
        )


        local_return = (
            total_return_between(
                local_rates_carry,
                start_date,
                latest_date
            )
        )


        fx_return = (
            total_return_between(
                fx_proxy_cad,
                start_date,
                latest_date
            )
        )


        canada_return = (
            total_return_between(
                canada_bonds,
                start_date,
                latest_date
            )
        )


        if (
            pd.isna(
                total_return
            )
            or pd.isna(
                local_return
            )
            or pd.isna(
                fx_return
            )
        ):

            interaction = (
                float("nan")
            )

        else:

            interaction = (
                total_return
                - local_return
                - fx_return
            )


        if (
            pd.isna(
                local_return
            )
            or pd.isna(
                canada_return
            )
        ):

            local_vs_canada = (
                float("nan")
            )

        else:

            local_vs_canada = (
                local_return
                - canada_return
            )


        return {

            "total":
                total_return,

            "local":
                local_return,

            "fx":
                fx_return,

            "interaction":
                interaction,

            "canada":
                canada_return,

            "local_vs_canada":
                local_vs_canada
        }


    period_attribution = {

        "YTD":
            calculate_period(
                ytd_start
            ),

        "1Y":
            calculate_period(
                one_year_start
            ),

        "3Y":
            calculate_period(
                three_year_start
            )
    }


    # --------------------------------------------------------
    # ROLLING 12-MONTH RETURNS
    # --------------------------------------------------------

    rolling_total = (
        rolling_total_return(
            em_unhedged_cad,
            252
        )
    )


    rolling_local = (
        rolling_total_return(
            local_rates_carry,
            252
        )
    )


    rolling_fx = (
        rolling_total_return(
            fx_proxy_cad,
            252
        )
    )


    rolling_canada = (
        rolling_total_return(
            canada_bonds,
            252
        )
    )


    rolling_interaction = (
        rolling_total
        - rolling_local
        - rolling_fx
    )


    rolling_local_vs_canada = (
        rolling_local
        - rolling_canada
    )


    rolling_frame = pd.DataFrame({

        "EM Unhedged CAD":
            rolling_total,

        "Local Rates & Carry":
            rolling_local,

        "LEMB-Matched FX":
            rolling_fx,

        "Interaction":
            rolling_interaction,

        "Synthetic Canada Federal":
            rolling_canada,

        "Local - Synthetic Canada":
            rolling_local_vs_canada

    }).dropna(
        how="all"
    )


    return {

        "em_unhedged_cad":
            em_unhedged_cad,

        "local_rates_carry":
            local_rates_carry,

        "fx_proxy_cad":
            fx_proxy_cad,

        "canada_bonds":
            canada_bonds,

        "local_vs_canada_index":
            local_vs_canada_index,

        "fx_impact_index":
            fx_impact_index,

        "period_attribution":
            period_attribution,

        "rolling":
            rolling_frame,

        "latest_date":
            latest_date,

        "currency_weights":
            currency_weights,

        "used_currency_weights":
            used_currency_weights,

        "fx_weight_coverage":
            fx_weight_coverage,

        "missing_fx_currencies":
            missing_fx_currencies,

        "holdings_as_of":
            holdings_as_of,

        "weight_source":
            weight_source,

        "canada_zfs_weight":
            CANADA_ZFS_WEIGHT,

        "canada_zfm_weight":
            CANADA_ZFM_WEIGHT,

        "canada_synthetic_duration":
            SYNTHETIC_CANADA_DURATION,

        "lemb_target_duration":
            LEMB_TARGET_DURATION
    }


EM_BOND_DECOMP = (
    build_em_local_bond_proxy_decomposition(
        prices
    )
)


# ============================================================
# PROMOTE EM BOND / FX SERIES INTO THE STANDARD SIGNAL ENGINE
# ============================================================
#
# These are not recomputed independently. They reuse the exact same
# series that appear on the EM decomposition page:
#
#   1) EM Local / Synthetic Canada
#      = Local Rates & Carry Proxy / Synthetic Canada Federal
#
#   2) LEMB-Matched EM FX / CAD
#      = the LEMB-weighted EM-currency basket expressed in CAD terms
#
# Once inserted into `ratios`, they receive the exact same 1Y/3Y/5Y/10Y
# averages, z-score, percentile, momentum, acceleration and individual
# indicator-page treatment as every other dashboard signal.

if EM_BOND_DECOMP is not None:

    ratios[
        "EM Local / Synthetic Canada"
    ] = (
        EM_BOND_DECOMP[
            "local_vs_canada_index"
        ]
    )

    ratios[
        "LEMB-Matched EM FX / CAD"
    ] = (
        EM_BOND_DECOMP[
            "fx_impact_index"
        ]
    )

    active_em_local_signals = (
        EM_LOCAL_SIGNALS.copy()
    )

else:

    active_em_local_signals = []


# ============================================================
# CALCULATE RESULTS
# ============================================================

results = []


for ratio_name in (
    PORTFOLIO_FACTORS
    + RISK_APPETITE
    + active_em_local_signals
):

    series = (
        ratios[ratio_name]
        .dropna()
    )


    avg_10y = trailing_average(
        series,
        10
    )

    avg_5y = trailing_average(
        series,
        5
    )

    avg_3y = trailing_average(
        series,
        3
    )

    avg_1y = trailing_average(
        series,
        1
    )


    current_ratio = latest_ratio(
        series
    )

    current_date = latest_date(
        series
    )


    # --------------------------------------------------------
    # DEVIATIONS
    # --------------------------------------------------------

    deviation_from_10y = (
        percentage_deviation(
            current_ratio,
            avg_10y
        )
    )

    deviation_from_5y = (
        percentage_deviation(
            current_ratio,
            avg_5y
        )
    )

    deviation_from_3y = (
        percentage_deviation(
            current_ratio,
            avg_3y
        )
    )

    deviation_from_1y = (
        percentage_deviation(
            current_ratio,
            avg_1y
        )
    )


    # --------------------------------------------------------
    # STRUCTURAL AVERAGES
    # --------------------------------------------------------

    one_vs_three_pct = (
        percentage_change(
            avg_1y,
            avg_3y
        )
    )

    three_vs_five_pct = (
        percentage_change(
            avg_3y,
            avg_5y
        )
    )

    five_vs_ten_pct = (
        percentage_change(
            avg_5y,
            avg_10y
        )
    )


    # --------------------------------------------------------
    # MOMENTUM
    #
    # 63-trading-day OLS slope of log(relative ratio),
    # annualized into approximate percentage terms.
    # --------------------------------------------------------

    (
        momentum_series,
        momentum_tstat_series
    ) = rolling_log_trend(
        series,
        window=MOMENTUM_WINDOW
    )

    valid_momentum = (
        momentum_series
        .dropna()
    )

    valid_tstats = (
        momentum_tstat_series
        .dropna()
    )

    if valid_momentum.empty:

        momentum_value = float("nan")

    else:

        momentum_value = (
            valid_momentum.iloc[-1]
        )

    if valid_tstats.empty:

        momentum_tstat = float("nan")

    else:

        momentum_tstat = (
            valid_tstats.iloc[-1]
        )


    # --------------------------------------------------------
    # ACCELERATION OF MOMENTUM
    #
    # Change in the annualized 63-day slope over 21 trading days.
    # This is a finite-difference approximation to a second
    # derivative of the log relative-performance ratio.
    # --------------------------------------------------------

    acceleration_series = (
        momentum_acceleration_series(
            momentum_series,
            lookback=ACCELERATION_LOOKBACK
        )
    )

    valid_acceleration = (
        acceleration_series
        .dropna()
    )

    if valid_acceleration.empty:

        acceleration_value = float("nan")

    else:

        acceleration_value = (
            valid_acceleration.iloc[-1]
        )

    acceleration_z = (
        trailing_zscore_of_series(
            acceleration_series,
            years=3
        )
    )


    # --------------------------------------------------------
    # POSITIONING
    # --------------------------------------------------------

    zscore_3y = (
        z_score(
            series,
            current_ratio,
            years=3
        )
    )


    percentile_3y = (
        historical_percentile(
            series,
            current_ratio,
            years=3
        )
    )


    positioning = (
        positioning_label(
            zscore_3y
        )
    )


    momentum = (
        momentum_label(
            momentum_value,
            momentum_tstat
        )
    )


    acceleration = (
        acceleration_label(
            acceleration_value,
            acceleration_z
        )
    )


    results.append({

        "bucket":
            indicator_buckets[
                ratio_name
            ],

        "ratio_name":
            ratio_name,

        "short_name":
            short_names[
                ratio_name
            ],

        "purpose":
            indicator_purposes[
                ratio_name
            ],

        "series":
            series,

        "avg_10y":
            avg_10y,

        "avg_5y":
            avg_5y,

        "avg_3y":
            avg_3y,

        "avg_1y":
            avg_1y,

        "current_ratio":
            current_ratio,

        "current_date":
            current_date,

        "deviation_from_10y":
            deviation_from_10y,

        "deviation_from_5y":
            deviation_from_5y,

        "deviation_from_3y":
            deviation_from_3y,

        "deviation_from_1y":
            deviation_from_1y,

        "one_vs_three_pct":
            one_vs_three_pct,

        "three_vs_five_pct":
            three_vs_five_pct,

        "five_vs_ten_pct":
            five_vs_ten_pct,

        "momentum_series":
            momentum_series,

        "momentum_tstat_series":
            momentum_tstat_series,

        "momentum_value":
            momentum_value,

        "momentum_tstat":
            momentum_tstat,

        "momentum":
            momentum,

        "acceleration_series":
            acceleration_series,

        "acceleration_value":
            acceleration_value,

        "acceleration_z":
            acceleration_z,

        "acceleration":
            acceleration,

        "zscore_3y":
            zscore_3y,

        "percentile_3y":
            percentile_3y,

        "positioning":
            positioning,

        "history_years":
            available_years(
                series
            )
    })


# ============================================================
# SHORT PAGE-1 INTERPRETATION
# ============================================================

def summary_interpretation(result):

    ratio_name = result[
        "ratio_name"
    ]

    z = result[
        "zscore_3y"
    ]

    momentum_value = result[
        "momentum_value"
    ]


    if pd.isna(z):
        return "No Signal"


    # The z-score determines which side has historical leadership.
    # Momentum is used only as a secondary directional cue.

    if ratio_name == "XDG.U / URTH":

        if z >= 1:
            return "Dividend Strong"

        elif z <= -1:
            return "Dividend Weak"

        elif momentum_value > 0:
            return "Dividend Improving"

        elif momentum_value < 0:
            return "Dividend Fading"

        else:
            return "Dividend Neutral"


    if ratio_name == "QUAL / PBUS":

        if z >= 1:
            return "Quality Strong"

        elif z <= -1:
            return "Quality Weak"

        elif momentum_value > 0:
            return "Quality Improving"

        elif momentum_value < 0:
            return "Quality Fading"

        else:
            return "Quality Neutral"


    if ratio_name == "50/50 IXC + MXI / URTH":

        if z >= 1:
            return "Resources Strong"

        elif z <= -1:
            return "Resources Weak"

        elif momentum_value > 0:
            return "Resources Improving"

        elif momentum_value < 0:
            return "Resources Fading"

        else:
            return "Resources Neutral"


    if ratio_name == "IJR / IWM":

        if z >= 1:
            return "Quality Smallcaps"

        elif z <= -1:
            return "Broad Smallcaps"

        elif momentum_value > 0:
            return "Quality Improving"

        elif momentum_value < 0:
            return "Broad Improving"

        else:
            return "Smallcaps Neutral"


    if ratio_name == "EMXC / MCHI":

        if z >= 1:
            return "Ex-China Leadership"

        elif z <= -1:
            return "China Leadership"

        elif momentum_value > 0:
            return "Ex-China Improving"

        elif momentum_value < 0:
            return "China Improving"

        else:
            return "Regional Neutral"


    if ratio_name == "IWF / IWD":

        if z >= 1:
            return "Growth Leadership"

        elif z <= -1:
            return "Value Leadership"

        elif momentum_value > 0:
            return "Growth Improving"

        elif momentum_value < 0:
            return "Value Improving"

        else:
            return "Style Neutral"


    if ratio_name == "IWC / IJR":

        if z >= 2:
            return "Speculation Extreme"

        elif z >= 1:
            return "Speculation Elevated"

        elif z <= -1:
            return "Speculation Weak"

        elif momentum_value > 0:
            return "Speculation Rising"

        elif momentum_value < 0:
            return "Speculation Fading"

        else:
            return "Speculation Neutral"


    if ratio_name == "KRE / SPY":

        if z >= 1:
            return "US Banks Strong"

        elif z <= -1:
            return "US Banks Weak"

        elif momentum_value > 0:
            return "Banks Improving"

        elif momentum_value < 0:
            return "Banks Fading"

        else:
            return "Banks Neutral"


    if ratio_name == "EXV1 / E600":

        if z >= 2:
            return "Europe Banks Extreme"

        elif z >= 1:
            return "Europe Banks Strong"

        elif z <= -1:
            return "Europe Banks Weak"

        elif momentum_value > 0:
            return "Banks Improving"

        elif momentum_value < 0:
            return "Banks Fading"

        else:
            return "Banks Neutral"


    if ratio_name == "SHYG / IGSB":

        if z >= 2:
            return "Credit Very Strong"

        elif z >= 1:
            return "Credit Strong"

        elif z <= -1:
            return "Credit Defensive"

        elif momentum_value > 0:
            return "Credit Improving"

        elif momentum_value < 0:
            return "Credit Fading"

        else:
            return "Credit Neutral"


    if ratio_name == "EM Local / Synthetic Canada":

        if z >= 1:
            return "EM Local Leadership"

        elif z <= -1:
            return "Canada Leadership"

        elif momentum_value > 0:
            return "EM Local Improving"

        elif momentum_value < 0:
            return "Canada Improving"

        else:
            return "Rates Relative Neutral"


    if ratio_name == "LEMB-Matched EM FX / CAD":

        if z >= 1:
            return "EM FX Strong"

        elif z <= -1:
            return "CAD Strong"

        elif momentum_value > 0:
            return "EM FX Strengthening"

        elif momentum_value < 0:
            return "EM FX Weakening"

        else:
            return "FX Neutral"


    return "Neutral"


# ============================================================
# CONCISE INDICATOR SMART COMMENTARY
# ============================================================

def smart_indicator_comment(result):

    percentile = result[
        "percentile_3y"
    ]

    z = result[
        "zscore_3y"
    ]

    momentum_value = result[
        "momentum_value"
    ]

    momentum_tstat = result[
        "momentum_tstat"
    ]

    acceleration_value = result[
        "acceleration_value"
    ]

    acceleration_z = result[
        "acceleration_z"
    ]

    ratio_name = result[
        "ratio_name"
    ]


    # --------------------------------------------------------
    # HISTORICAL POSITIONING
    # --------------------------------------------------------

    if (
        pd.isna(percentile)
        or pd.isna(z)
    ):

        positioning_text = (
            "Historical positioning is unavailable."
        )

    elif percentile >= 95:

        positioning_text = (
            f"The ratio is historically elevated at the "
            f"{percentile:.1f}th percentile of its 3-year "
            f"range (z-score {z:+.2f})."
        )

    elif percentile >= 75:

        positioning_text = (
            f"The ratio is above its recent historical norm, "
            f"at the {percentile:.1f}th percentile "
            f"(z-score {z:+.2f})."
        )

    elif percentile <= 5:

        positioning_text = (
            f"The ratio is historically depressed at the "
            f"{percentile:.1f}th percentile of its 3-year "
            f"range (z-score {z:+.2f})."
        )

    elif percentile <= 25:

        positioning_text = (
            f"The ratio is below its recent historical norm, "
            f"at the {percentile:.1f}th percentile "
            f"(z-score {z:+.2f})."
        )

    else:

        positioning_text = (
            f"The ratio is near its recent historical norm, "
            f"at the {percentile:.1f}th percentile "
            f"(z-score {z:+.2f})."
        )


    # --------------------------------------------------------
    # MOMENTUM + ACCELERATION
    # --------------------------------------------------------

    if (
        pd.isna(momentum_value)
        or pd.isna(momentum_tstat)
    ):

        momentum_text = (
            "Trend momentum is unavailable."
        )

    else:

        direction = (
            "positive"
            if momentum_value > 0
            else "negative"
            if momentum_value < 0
            else "flat"
        )

        momentum_text = (
            f"63-day trend momentum is {direction} at "
            f"{momentum_value:+.1f}% annualized "
            f"(slope t-stat {momentum_tstat:+.2f})."
        )


    if (
        pd.isna(acceleration_value)
        or pd.isna(acceleration_z)
    ):

        acceleration_text = (
            " Momentum acceleration is unavailable."
        )

    else:

        acceleration_direction = (
            "accelerating"
            if acceleration_value > 0
            else "decelerating"
            if acceleration_value < 0
            else "stable"
        )

        acceleration_text = (
            f" Momentum is {acceleration_direction}: "
            f"{acceleration_value:+.1f} percentage points "
            f"versus 21 trading days ago "
            f"(3Y acceleration z-score {acceleration_z:+.2f})."
        )


    # --------------------------------------------------------
    # INDICATOR-SPECIFIC IMPLICATION
    # --------------------------------------------------------

    if ratio_name == "XDG.U / URTH":

        implication = (
            "A rising ratio favors global high-dividend leadership; "
            "a falling ratio favors the broader world market."
        )


    elif ratio_name == "QUAL / PBUS":

        implication = (
            "A rising ratio favors US quality leadership; "
            "a falling ratio favors the broader US market."
        )


    elif ratio_name == "50/50 IXC + MXI / URTH":

        implication = (
            "A rising ratio favors Energy and Materials leadership "
            "relative to global equities."
        )


    elif ratio_name == "IJR / IWM":

        implication = (
            "A rising ratio favors profitable small caps; "
            "a falling ratio favors the broader small-cap universe."
        )


    elif ratio_name == "EMXC / MCHI":

        implication = (
            "A rising ratio favors EM ex-China; "
            "a falling ratio favors China."
        )


    elif ratio_name == "IWF / IWD":

        implication = (
            "A rising ratio favors Growth; "
            "a falling ratio favors Value."
        )


    elif ratio_name == "IWC / IJR":

        implication = (
            "A rising ratio is consistent with stronger "
            "small-cap speculative appetite."
        )


    elif ratio_name == "KRE / SPY":

        implication = (
            "A rising ratio indicates stronger US regional-bank "
            "leadership relative to the broad market."
        )


    elif ratio_name == "EXV1 / E600":

        implication = (
            "A rising ratio indicates stronger European-bank "
            "leadership relative to Europe."
        )


    elif ratio_name == "SHYG / IGSB":

        implication = (
            "A rising ratio indicates stronger appetite for "
            "lower-quality corporate credit."
        )


    elif ratio_name == "EM Local / Synthetic Canada":

        implication = (
            "A rising ratio indicates the EM local rates & carry proxy "
            "is outperforming the duration-matched Canadian federal benchmark."
        )


    elif ratio_name == "LEMB-Matched EM FX / CAD":

        implication = (
            "A rising index indicates LEMB-weighted EM currencies are "
            "appreciating versus CAD; a falling index indicates CAD strength."
        )


    else:

        implication = ""


    return (
        positioning_text
        + " "
        + momentum_text
        + acceleration_text
        + " "
        + implication
    )


# ============================================================
# SMART HEADLINE
# ============================================================

def smart_headline(result):

    momentum = result[
        "momentum"
    ]

    acceleration = result[
        "acceleration"
    ]


    if momentum in [
        "Strong Positive",
        "Positive"
    ]:

        if acceleration in [
            "Strongly Accelerating",
            "Accelerating"
        ]:
            return "Positive Momentum, Accelerating"

        if acceleration in [
            "Strongly Decelerating",
            "Decelerating"
        ]:
            return "Positive Momentum, Decelerating"

        return "Positive Momentum"


    if momentum in [
        "Strong Negative",
        "Negative"
    ]:

        if acceleration in [
            "Strongly Decelerating",
            "Decelerating"
        ]:
            return "Negative Momentum, Worsening"

        if acceleration in [
            "Strongly Accelerating",
            "Accelerating"
        ]:
            return "Negative Momentum, Recovering"

        return "Negative Momentum"


    if momentum == "Flat / Noisy":

        if acceleration in [
            "Strongly Accelerating",
            "Accelerating"
        ]:
            return "Flat Trend, Improving"

        if acceleration in [
            "Strongly Decelerating",
            "Decelerating"
        ]:
            return "Flat Trend, Weakening"

        return "Flat / Noisy Trend"


    return "Signal Unavailable"


# ============================================================
# PRE-CALCULATE COMMENTS
# ============================================================

for result in results:

    result[
        "smart_headline"
    ] = smart_headline(
        result
    )

    result[
        "smart_comment"
    ] = smart_indicator_comment(
        result
    )

    result[
        "summary_interpretation"
    ] = summary_interpretation(
        result
    )


# ============================================================
# BUCKET / OVERALL COMMENTARY
# ============================================================

def get_results_by_bucket(bucket):

    return [

        result

        for result in results

        if result["bucket"] == bucket
    ]


def strongest_and_weakest(
    bucket_results
):

    valid = [

        result

        for result in bucket_results

        if not pd.isna(
            result["zscore_3y"]
        )
    ]


    if not valid:
        return None, None


    strongest = max(
        valid,
        key=lambda x: x["zscore_3y"]
    )


    weakest = min(
        valid,
        key=lambda x: x["zscore_3y"]
    )


    return strongest, weakest


def portfolio_factor_summary():

    bucket = get_results_by_bucket(
        "Portfolio Factor Positioning"
    )


    positive = [
        x
        for x in bucket
        if x["zscore_3y"] >= 1
    ]

    negative = [
        x
        for x in bucket
        if x["zscore_3y"] <= -1
    ]


    strongest, weakest = (
        strongest_and_weakest(
            bucket
        )
    )


    if positive and negative:

        opening = (
            "Portfolio factor positioning is divergent rather "
            "than showing a single dominant style regime."
        )


    elif len(positive) >= 2:

        opening = (
            "Portfolio factor positioning shows relatively broad "
            "positive leadership across several monitored factors."
        )


    elif len(negative) >= 2:

        opening = (
            "Portfolio factor positioning is generally soft, "
            "with several monitored factors below their recent "
            "historical norms."
        )


    else:

        opening = (
            "Portfolio factor positioning is broadly mixed and "
            "close to historical norms."
        )


    detail = ""


    if (
        strongest is not None
        and weakest is not None
    ):

        detail = (

            f" The strongest relative positioning signal is "
            f"{strongest['short_name']} "
            f"(z-score {strongest['zscore_3y']:+.2f}, "
            f"{strongest['percentile_3y']:.1f}th percentile), "
            f"while {weakest['short_name']} is the weakest "
            f"(z-score {weakest['zscore_3y']:+.2f}, "
            f"{weakest['percentile_3y']:.1f}th percentile)."
        )


    positive_momentum = [
        x
        for x in bucket
        if x["momentum"] in [
            "Strong Positive",
            "Positive"
        ]
    ]


    negative_momentum = [
        x
        for x in bucket
        if x["momentum"] in [
            "Strong Negative",
            "Negative"
        ]
    ]


    momentum_text = ""


    if positive_momentum:

        momentum_text += (
            " Positive trend momentum is visible in "
            + ", ".join(
                x["short_name"]
                for x in positive_momentum
            )
            + "."
        )


    if negative_momentum:

        momentum_text += (
            " Negative trend momentum is visible in "
            + ", ".join(
                x["short_name"]
                for x in negative_momentum
            )
            + "."
        )


    return (
        opening
        + detail
        + momentum_text
    )


def risk_appetite_summary():

    bucket = get_results_by_bucket(
        "Risk Appetite / Speculation"
    )


    high = [
        x
        for x in bucket
        if x["zscore_3y"] >= 1
    ]


    low = [
        x
        for x in bucket
        if x["zscore_3y"] <= -1
    ]


    neutral = [
        x
        for x in bucket
        if -1 < x["zscore_3y"] < 1
    ]


    if len(high) >= 3:

        opening = (
            "Risk appetite is broadly elevated, with multiple "
            "independent indicators above their recent historical norms."
        )


    elif len(high) == 2:

        opening = (
            "Risk appetite is selectively elevated rather than "
            "uniformly strong across all monitored markets."
        )


    elif high and low:

        opening = (
            "Risk signals are fragmented, with meaningful "
            "disagreement across the monitored indicators."
        )


    elif len(low) >= 3:

        opening = (
            "Risk appetite is broadly defensive across the "
            "monitored indicators."
        )


    else:

        opening = (
            "Risk appetite is broadly neutral, with limited "
            "cross-signal confirmation."
        )


    high_text = ""


    if high:

        high_text = (
            " Elevated positioning is currently visible in "
            + ", ".join(
                x["short_name"]
                for x in high
            )
            + "."
        )


    low_text = ""


    if low:

        low_text = (
            " Depressed positioning is currently visible in "
            + ", ".join(
                x["short_name"]
                for x in low
            )
            + "."
        )


    accelerating = [
        x
        for x in bucket
        if x["acceleration"] in [
            "Strongly Accelerating",
            "Accelerating"
        ]
    ]


    decelerating = [
        x
        for x in bucket
        if x["acceleration"] in [
            "Strongly Decelerating",
            "Decelerating"
        ]
    ]


    acceleration_text = ""


    if accelerating:

        acceleration_text += (
            " Momentum is accelerating in "
            + ", ".join(
                x["short_name"]
                for x in accelerating
            )
            + "."
        )


    if decelerating:

        acceleration_text += (
            " Momentum is decelerating in "
            + ", ".join(
                x["short_name"]
                for x in decelerating
            )
            + "."
        )


    return (
        opening
        + high_text
        + low_text
        + acceleration_text
    )


def overall_market_summary():

    risk_bucket = get_results_by_bucket(
        "Risk Appetite / Speculation"
    )


    factor_bucket = get_results_by_bucket(
        "Portfolio Factor Positioning"
    )


    risk_high_count = sum(
        x["zscore_3y"] >= 1
        for x in risk_bucket
    )


    risk_low_count = sum(
        x["zscore_3y"] <= -1
        for x in risk_bucket
    )


    factor_high_count = sum(
        x["zscore_3y"] >= 1
        for x in factor_bucket
    )


    factor_low_count = sum(
        x["zscore_3y"] <= -1
        for x in factor_bucket
    )


    if (
        risk_high_count >= 3
        and risk_low_count == 0
    ):

        opening = (
            "The cross-asset dashboard currently shows broad "
            "evidence of elevated risk appetite."
        )


    elif risk_high_count == 2:

        opening = (
            "The dashboard points to selectively elevated risk "
            "appetite rather than a uniformly strong risk-seeking "
            "regime."
        )


    elif (
        risk_high_count > 0
        and risk_low_count > 0
    ):

        opening = (
            "The risk backdrop is fragmented, with meaningful "
            "divergence across the monitored markets."
        )


    elif risk_low_count >= 3:

        opening = (
            "The dashboard currently points to broadly defensive "
            "risk positioning."
        )


    else:

        opening = (
            "The overall risk backdrop is relatively balanced, "
            "with limited confirmation of a strong directional "
            "risk regime."
        )


    if (
        factor_high_count > 0
        and factor_low_count > 0
    ):

        factor_text = (
            " Portfolio factor leadership is simultaneously "
            "divergent, suggesting that the underlying equity "
            "style environment is mixed."
        )


    elif factor_low_count >= 2:

        factor_text = (
            " Portfolio factor positioning remains relatively soft."
        )


    elif factor_high_count >= 2:

        factor_text = (
            " Portfolio factors also show relatively broad "
            "positive leadership."
        )


    else:

        factor_text = (
            " Portfolio-factor positioning remains mostly neutral."
        )


    positive_risk_momentum = sum(
        x["momentum"] in [
            "Strong Positive",
            "Positive"
        ]
        for x in risk_bucket
    )


    negative_risk_momentum = sum(
        x["momentum"] in [
            "Strong Negative",
            "Negative"
        ]
        for x in risk_bucket
    )


    if positive_risk_momentum >= 3:

        momentum_text = (
            " Trend momentum is positive across most risk indicators."
        )


    elif negative_risk_momentum >= 3:

        momentum_text = (
            " Trend momentum is negative across most risk indicators."
        )


    else:

        momentum_text = (
            " Trend momentum remains mixed across the risk signals."
        )


    return (
        opening
        + factor_text
        + momentum_text
    )


PORTFOLIO_COMMENT = (
    portfolio_factor_summary()
)

RISK_COMMENT = (
    risk_appetite_summary()
)

OVERALL_COMMENT = (
    overall_market_summary()
)


# ============================================================
# PDF HELPERS
# ============================================================

def make_text_panel(
    fig,
    rect
):

    ax = fig.add_axes(
        rect
    )

    ax.axis(
        "off"
    )

    return ax


def add_mcgill_banner(fig):

    banner = Rectangle(
        (0, 0.955),
        1,
        0.045,
        transform=fig.transFigure,
        facecolor=BANNER_COLOR,
        edgecolor="none",
        zorder=20
    )

    fig.patches.append(
        banner
    )


    fig.text(
        0.055,
        0.9775,
        REPORT_BANNER,
        fontsize=10,
        fontweight="bold",
        color=BANNER_TEXT_COLOR,
        ha="left",
        va="center",
        zorder=21
    )


    fig.text(
        0.945,
        0.9775,
        REPORT_TITLE,
        fontsize=8.3,
        color=BANNER_TEXT_COLOR,
        ha="right",
        va="center",
        zorder=21
    )


def add_page_footer(
    fig,
    page_number
):

    fig.text(
        0.055,
        0.022,
        REPORT_BANNER,
        fontsize=7.2,
        alpha=0.65,
        ha="left",
        va="bottom"
    )


    fig.text(
        0.945,
        0.022,
        f"Page {page_number}",
        fontsize=7.2,
        alpha=0.65,
        ha="right",
        va="bottom"
    )


def style_table(
    table,
    body_fontsize=8.2
):

    table.auto_set_font_size(
        False
    )

    for (
        row,
        col
    ), cell in table.get_celld().items():

        cell.set_linewidth(
            0.55
        )

        if row == 0:

            cell.set_facecolor(
                HEADER_BG
            )

            cell.set_text_props(
                weight="bold",
                fontsize=8.2
            )

        else:

            cell.set_text_props(
                fontsize=body_fontsize
            )


# ============================================================
# TABLE CONDITIONAL FORMATTING
# ============================================================

def apply_summary_table_colors(
    table,
    table_results
):

    for row_index, result in enumerate(
        table_results,
        start=1
    ):

        z = result[
            "zscore_3y"
        ]

        percentile = result[
            "percentile_3y"
        ]

        momentum = result[
            "momentum"
        ]

        acceleration = result[
            "acceleration"
        ]

        positioning = result[
            "positioning"
        ]


        # Z-SCORE

        if pd.isna(z):

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT

        elif z >= 1:

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif z <= -1:

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 1)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 1)
        ].get_text().set_color(
            fg
        )


        # PERCENTILE

        if pd.isna(percentile):

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT

        elif percentile >= 75:

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif percentile <= 25:

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 2)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 2)
        ].get_text().set_color(
            fg
        )


        # MOMENTUM

        if momentum in [
            "Strong Positive",
            "Positive"
        ]:

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif momentum in [
            "Strong Negative",
            "Negative"
        ]:

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 3)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 3)
        ].get_text().set_color(
            fg
        )


        # ACCELERATION

        if acceleration in [
            "Strongly Accelerating",
            "Accelerating"
        ]:

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif acceleration in [
            "Strongly Decelerating",
            "Decelerating"
        ]:

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 4)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 4)
        ].get_text().set_color(
            fg
        )


        # POSITIONING

        if positioning in [
            "High",
            "Extremely High"
        ]:

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif positioning in [
            "Low",
            "Extremely Low"
        ]:

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 5)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 5)
        ].get_text().set_color(
            fg
        )


        # INTERPRETATION

        interpretation = (
            result[
                "summary_interpretation"
            ]
        )


        if any(
            word in interpretation
            for word in [
                "Strong",
                "Elevated",
                "Extreme",
                "Improving",
                "Rising"
            ]
        ):

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT


        elif any(
            word in interpretation
            for word in [
                "Weak",
                "Defensive",
                "Fading"
            ]
        ):

            bg = CELL_LOW
            fg = CELL_LOW_TEXT


        elif interpretation in [
            "Broad Smallcaps",
            "Quality Smallcaps",
            "Broad Improving",
            "Ex-China Leadership",
            "China Leadership",
            "Ex-China Improving",
            "China Improving",
            "Growth Leadership",
            "Value Leadership",
            "Growth Improving",
            "Value Improving"
        ]:

            bg = CELL_WATCH
            fg = CELL_WATCH_TEXT


        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 6)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 6)
        ].get_text().set_color(
            fg
        )

        table[
            (row_index, 6)
        ].get_text().set_fontweight(
            "bold"
        )



# ============================================================
# PAGE 1 - EXECUTIVE DASHBOARD HELPERS
# ============================================================

SUMMARY_BORDER = "#D9DDE3"
SUMMARY_TEXT = "#1F2933"
SUMMARY_MUTED = "#6B7280"
SUMMARY_SOFT = "#F7F8FA"
SUMMARY_HEADER = "#F2F4F7"
SUMMARY_ROW_ALT = "#FBFCFD"


def style_executive_summary_table(
    table,
    table_results
):
    """
    Professional first-page table styling:
    - light header
    - restrained horizontal rules
    - subtle alternating rows
    - conditional color only where it adds signal
    """

    table.auto_set_font_size(
        False
    )

    n_rows = len(
        table_results
    )

    n_cols = 7

    # --------------------------------------------------------
    # BASE STYLE
    # --------------------------------------------------------

    for (
        row,
        col
    ), cell in table.get_celld().items():

        cell.set_edgecolor(
            SUMMARY_BORDER
        )

        if row == 0:

            cell.visible_edges = "B"

            cell.set_linewidth(
                0.9
            )

            cell.set_facecolor(
                SUMMARY_HEADER
            )

            cell.set_text_props(
                fontsize=7.1,
                weight="bold",
                color="#4B5563"
            )

            cell.PAD = 0.018

        else:

            cell.visible_edges = "B"

            cell.set_linewidth(
                0.45
            )

            cell.set_facecolor(
                "white"
                if row % 2 == 1
                else SUMMARY_ROW_ALT
            )

            cell.set_text_props(
                fontsize=7.15,
                color=SUMMARY_TEXT
            )

            cell.PAD = 0.020


    # --------------------------------------------------------
    # ALIGNMENT / EMPHASIS
    # --------------------------------------------------------

    for row in range(
        1,
        n_rows + 1
    ):

        table[
            (row, 0)
        ].get_text().set_ha(
            "left"
        )

        table[
            (row, 0)
        ].get_text().set_fontweight(
            "bold"
        )

        table[
            (row, 0)
        ].get_text().set_fontsize(
            6.9
        )

        table[
            (row, 3)
        ].get_text().set_fontsize(
            6.65
        )

        table[
            (row, 4)
        ].get_text().set_fontsize(
            6.35
        )

        table[
            (row, 5)
        ].get_text().set_fontsize(
            6.75
        )

        table[
            (row, 6)
        ].get_text().set_fontsize(
            6.75
        )

        table[
            (row, 6)
        ].get_text().set_fontweight(
            "bold"
        )


    # --------------------------------------------------------
    # SIGNAL CELL COLORS
    # --------------------------------------------------------

    for row_index, result in enumerate(
        table_results,
        start=1
    ):

        momentum = result[
            "momentum"
        ]

        acceleration = result[
            "acceleration"
        ]

        positioning = result[
            "positioning"
        ]

        interpretation = result[
            "summary_interpretation"
        ]


        # Momentum
        if momentum in [
            "Strong Positive",
            "Positive"
        ]:

            table[
                (row_index, 3)
            ].set_facecolor(
                CELL_HIGH
            )

            table[
                (row_index, 3)
            ].get_text().set_color(
                CELL_HIGH_TEXT
            )

        elif momentum in [
            "Strong Negative",
            "Negative"
        ]:

            table[
                (row_index, 3)
            ].set_facecolor(
                CELL_LOW
            )

            table[
                (row_index, 3)
            ].get_text().set_color(
                CELL_LOW_TEXT
            )

        else:

            table[
                (row_index, 3)
            ].set_facecolor(
                CELL_NEUTRAL
            )

            table[
                (row_index, 3)
            ].get_text().set_color(
                CELL_NEUTRAL_TEXT
            )


        # Acceleration
        if acceleration in [
            "Strongly Accelerating",
            "Accelerating"
        ]:

            table[
                (row_index, 4)
            ].set_facecolor(
                CELL_HIGH
            )

            table[
                (row_index, 4)
            ].get_text().set_color(
                CELL_HIGH_TEXT
            )

        elif acceleration in [
            "Strongly Decelerating",
            "Decelerating"
        ]:

            table[
                (row_index, 4)
            ].set_facecolor(
                CELL_LOW
            )

            table[
                (row_index, 4)
            ].get_text().set_color(
                CELL_LOW_TEXT
            )

        else:

            table[
                (row_index, 4)
            ].set_facecolor(
                CELL_NEUTRAL
            )

            table[
                (row_index, 4)
            ].get_text().set_color(
                CELL_NEUTRAL_TEXT
            )


        # Positioning
        if positioning in [
            "High",
            "Extremely High"
        ]:

            table[
                (row_index, 5)
            ].set_facecolor(
                CELL_HIGH
            )

            table[
                (row_index, 5)
            ].get_text().set_color(
                CELL_HIGH_TEXT
            )

        elif positioning in [
            "Low",
            "Extremely Low"
        ]:

            table[
                (row_index, 5)
            ].set_facecolor(
                CELL_LOW
            )

            table[
                (row_index, 5)
            ].get_text().set_color(
                CELL_LOW_TEXT
            )

        else:

            table[
                (row_index, 5)
            ].set_facecolor(
                CELL_NEUTRAL
            )

            table[
                (row_index, 5)
            ].get_text().set_color(
                CELL_NEUTRAL_TEXT
            )


        # Interpretation: directional/style signals are watch-toned,
        # explicitly strong/weak signals retain positive/negative colors.
        if any(
            word in interpretation
            for word in [
                "Strong",
                "Elevated",
                "Extreme",
                "Rising"
            ]
        ):

            bg = CELL_HIGH
            fg = CELL_HIGH_TEXT

        elif any(
            word in interpretation
            for word in [
                "Weak",
                "Defensive",
                "Fading"
            ]
        ):

            bg = CELL_LOW
            fg = CELL_LOW_TEXT

        elif any(
            word in interpretation
            for word in [
                "Leadership",
                "Improving",
                "Smallcaps"
            ]
        ):

            bg = CELL_WATCH
            fg = CELL_WATCH_TEXT

        else:

            bg = CELL_NEUTRAL
            fg = CELL_NEUTRAL_TEXT


        table[
            (row_index, 6)
        ].set_facecolor(
            bg
        )

        table[
            (row_index, 6)
        ].get_text().set_color(
            fg
        )



# ============================================================
# INTERNAL PDF HYPERLINK HELPERS
# ============================================================


def style_clickable_indicator_names(
    table,
    table_results
):
    """
    Make page-1 indicator-name cells look like hyperlinks.

    The first-column text is:
        - conventional hyperlink blue
        - underlined
        - bold enough to remain legible in the summary table

    The actual clickable PDF area is added separately by
    capture_indicator_cell_links() + pypdf post-processing.
    """

    for row_index, result in enumerate(
        table_results,
        start=1
    ):

        cell = table[
            (
                row_index,
                0
            )
        ]

        cell_text = (
            cell.get_text()
        )

        cell_text.set_color(
            HYPERLINK_COLOR
        )

        cell_text.set_fontweight(
            "bold"
        )

        # Matplotlib Text supports underline in current releases.
        # If a very old Matplotlib build is used, this line can simply
        # be removed and the blue text still communicates clickability.
        try:
            cell_text.set_underline(
                True
            )
        except AttributeError:
            pass


def capture_indicator_cell_links(
    fig,
    table,
    table_results,
    link_rectangles
):
    """
    Capture the exact first-column cell rectangles from a rendered
    Matplotlib table and convert them into PDF-point coordinates.

    These rectangles are later used as invisible clickable areas on
    page 1 of the finished PDF.
    """

    fig.canvas.draw()

    renderer = (
        fig.canvas.get_renderer()
    )

    inverse_figure_transform = (
        fig.transFigure.inverted()
    )

    page_width_points = (
        PAGE_SIZE[0]
        * 72.0
    )

    page_height_points = (
        PAGE_SIZE[1]
        * 72.0
    )


    for row_index, result in enumerate(
        table_results,
        start=1
    ):

        cell = table[
            (
                row_index,
                0
            )
        ]

        bbox_display = (
            cell.get_window_extent(
                renderer=renderer
            )
        )

        lower_left = (
            inverse_figure_transform.transform(
                (
                    bbox_display.x0,
                    bbox_display.y0
                )
            )
        )

        upper_right = (
            inverse_figure_transform.transform(
                (
                    bbox_display.x1,
                    bbox_display.y1
                )
            )
        )


        # Slightly inset the clickable rectangle so adjacent cell
        # boundaries do not accidentally become part of the link.
        inset_x = (
            0.002
            * page_width_points
        )

        inset_y = (
            0.001
            * page_height_points
        )


        link_rectangles[
            result[
                "ratio_name"
            ]
        ] = (
            lower_left[0]
            * page_width_points
            + inset_x,

            lower_left[1]
            * page_height_points
            + inset_y,

            upper_right[0]
            * page_width_points
            - inset_x,

            upper_right[1]
            * page_height_points
            - inset_y
        )


def add_internal_indicator_links(
    pdf_path,
    link_rectangles,
    indicator_page_numbers
):
    """
    Add invisible internal PDF hyperlinks from the indicator-name cells
    on page 1 to the corresponding detailed indicator pages.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Finished PDF generated by Matplotlib.

    link_rectangles : dict
        ratio_name -> (x0, y0, x1, y1) in PDF points on page 1.

    indicator_page_numbers : dict
        ratio_name -> human-readable 1-based PDF page number.
    """

    pdf_path = Path(
        pdf_path
    )

    if (
        not pdf_path.exists()
        or not link_rectangles
        or not indicator_page_numbers
    ):

        return


    temp_path = (
        pdf_path.parent
        / (
            pdf_path.stem
            + "_linked_tmp.pdf"
        )
    )


    with open(
        pdf_path,
        "rb"
    ) as source_file:

        reader = PdfReader(
            source_file
        )

        writer = PdfWriter()


        for page in reader.pages:

            writer.add_page(
                page
            )


        if len(
            writer.pages
        ) == 0:

            return


        for ratio_name, rect in (
            link_rectangles.items()
        ):

            if ratio_name not in (
                indicator_page_numbers
            ):

                continue


            # The dashboard uses 1-based page numbers for display;
            # pypdf uses 0-based page indexes.
            target_page_index = (
                indicator_page_numbers[
                    ratio_name
                ]
                - 1
            )


            if not (
                0
                <= target_page_index
                < len(
                    writer.pages
                )
            ):

                continue


            link = Link(
                rect=rect,
                target_page_index=
                    target_page_index,
                fit=Fit.fit(),
                border=[
                    0,
                    0,
                    0
                ]
            )


            writer.add_annotation(
                page_number=0,
                annotation=link
            )


        with open(
            temp_path,
            "wb"
        ) as output_file:

            writer.write(
                output_file
            )


    temp_path.replace(
        pdf_path
    )


# ============================================================
# CREATE PDF
# ============================================================

page_number = 1

# Page-1 clickable indicator cells and their detailed-page destinations.
summary_link_rectangles = {}
indicator_page_numbers = {}


with PdfPages(
    OUTPUT_PDF
) as pdf:


    # ========================================================
    # PAGE 1 - SUMMARY DASHBOARD
    # ========================================================

    fig = plt.figure(
        figsize=PAGE_SIZE,
        facecolor="white"
    )


    add_mcgill_banner(
        fig
    )


    # --------------------------------------------------------
    # TITLE / DATE
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.885,
        REPORT_TITLE,
        fontsize=20,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.055,
        0.852,
        "Cross-asset positioning, trend momentum and momentum acceleration",
        fontsize=8.3,
        color=SUMMARY_MUTED,
        ha="left",
        va="center"
    )


    fig.text(
        0.945,
        0.885,
        "AS OF",
        fontsize=6.8,
        fontweight="bold",
        color=SUMMARY_MUTED,
        ha="right",
        va="center"
    )


    fig.text(
        0.945,
        0.854,
        today.strftime("%Y-%m-%d"),
        fontsize=9.4,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="right",
        va="center"
    )


    # --------------------------------------------------------
    # RESULT GROUPS
    # --------------------------------------------------------

    portfolio_results = [

        result

        for result in results

        if (
            result["bucket"]
            == "Portfolio Factor Positioning"
        )
    ]


    risk_results = [

        result

        for result in results

        if (
            result["bucket"]
            == "Risk Appetite / Speculation"
        )
    ]


    em_local_results = [

        result

        for result in results

        if (
            result["bucket"]
            == "EM Local Bond / Currency"
        )
    ]


    # --------------------------------------------------------
    # PORTFOLIO FACTORS
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.792,
        "Portfolio Factor Positioning",
        fontsize=11.4,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.945,
        0.792,
        f"{len(portfolio_results)} indicators",
        fontsize=7.0,
        color=SUMMARY_MUTED,
        ha="right",
        va="center"
    )


    fig.lines.append(
        plt.Line2D(
            [0.055, 0.945],
            [0.775, 0.775],
            transform=fig.transFigure,
            color=BANNER_COLOR,
            linewidth=1.25,
            alpha=0.90
        )
    )


    portfolio_rows = [

        [
            result["short_name"],

            format_signed(
                result["zscore_3y"]
            ),

            format_percent(
                result["percentile_3y"]
            ),

            result["momentum"],

            result["acceleration"],

            result["positioning"],

            result["summary_interpretation"]
        ]

        for result in portfolio_results
    ]


    ax_portfolio = fig.add_axes([
        0.055,
        0.550,
        0.89,
        0.210
    ])

    ax_portfolio.axis(
        "off"
    )


    portfolio_table = ax_portfolio.table(

        cellText=portfolio_rows,

        colLabels=[
            "Indicator",
            "3Y Z",
            "Percentile",
            "Momentum",
            "Acceleration",
            "Positioning",
            "Interpretation"
        ],

        colWidths=[
            0.230,
            0.075,
            0.090,
            0.135,
            0.155,
            0.120,
            0.195
        ],

        bbox=[
            0,
            0,
            1,
            1
        ],

        cellLoc="center",
        colLoc="center"
    )


    style_executive_summary_table(
        portfolio_table,
        portfolio_results
    )


    style_clickable_indicator_names(
        portfolio_table,
        portfolio_results
    )


    capture_indicator_cell_links(
        fig,
        portfolio_table,
        portfolio_results,
        summary_link_rectangles
    )


    # --------------------------------------------------------
    # RISK APPETITE
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.505,
        "Risk Appetite / Speculation",
        fontsize=11.4,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.945,
        0.505,
        f"{len(risk_results)} indicators",
        fontsize=7.0,
        color=SUMMARY_MUTED,
        ha="right",
        va="center"
    )


    fig.lines.append(
        plt.Line2D(
            [0.055, 0.945],
            [0.488, 0.488],
            transform=fig.transFigure,
            color=BANNER_COLOR,
            linewidth=1.25,
            alpha=0.90
        )
    )


    risk_rows = [

        [
            result["short_name"],

            format_signed(
                result["zscore_3y"]
            ),

            format_percent(
                result["percentile_3y"]
            ),

            result["momentum"],

            result["acceleration"],

            result["positioning"],

            result["summary_interpretation"]
        ]

        for result in risk_results
    ]


    ax_risk = fig.add_axes([
        0.055,
        0.350,
        0.89,
        0.125
    ])

    ax_risk.axis(
        "off"
    )


    risk_table = ax_risk.table(

        cellText=risk_rows,

        colLabels=[
            "Indicator",
            "3Y Z",
            "Percentile",
            "Momentum",
            "Acceleration",
            "Positioning",
            "Interpretation"
        ],

        colWidths=[
            0.230,
            0.075,
            0.090,
            0.135,
            0.155,
            0.120,
            0.195
        ],

        bbox=[
            0,
            0,
            1,
            1
        ],

        cellLoc="center",
        colLoc="center"
    )


    style_executive_summary_table(
        risk_table,
        risk_results
    )


    style_clickable_indicator_names(
        risk_table,
        risk_results
    )


    capture_indicator_cell_links(
        fig,
        risk_table,
        risk_results,
        summary_link_rectangles
    )


    # --------------------------------------------------------
    # EM LOCAL BOND / CURRENCY SIGNALS
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.305,
        "EM Local Bond / Currency",
        fontsize=11.0,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.945,
        0.305,
        f"{len(em_local_results)} indicators",
        fontsize=7.0,
        color=SUMMARY_MUTED,
        ha="right",
        va="center"
    )


    fig.lines.append(
        plt.Line2D(
            [0.055, 0.945],
            [0.288, 0.288],
            transform=fig.transFigure,
            color=BANNER_COLOR,
            linewidth=1.25,
            alpha=0.90
        )
    )


    em_local_rows = [

        [
            (
                "EM Local Bonds vs Canada"
                if result["ratio_name"] == "EM Local / Synthetic Canada"
                else result["short_name"]
            ),

            format_signed(
                result["zscore_3y"]
            ),

            format_percent(
                result["percentile_3y"]
            ),

            result["momentum"],

            result["acceleration"],

            result["positioning"],

            result["summary_interpretation"]
        ]

        for result in em_local_results
    ]


    ax_em_local = fig.add_axes([
        0.055,
        0.195,
        0.89,
        0.080
    ])

    ax_em_local.axis(
        "off"
    )


    em_local_table = ax_em_local.table(

        cellText=em_local_rows,

        colLabels=[
            "Indicator",
            "3Y Z",
            "Percentile",
            "Momentum",
            "Acceleration",
            "Positioning",
            "Interpretation"
        ],

        colWidths=[
            0.230,
            0.075,
            0.090,
            0.135,
            0.155,
            0.120,
            0.195
        ],

        bbox=[
            0,
            0,
            1,
            1
        ],

        cellLoc="center",
        colLoc="center"
    )


    style_executive_summary_table(
        em_local_table,
        em_local_results
    )


    style_clickable_indicator_names(
        em_local_table,
        em_local_results
    )


    capture_indicator_cell_links(
        fig,
        em_local_table,
        em_local_results,
        summary_link_rectangles
    )


    # --------------------------------------------------------
    # COMPACT METHODOLOGY STRIP
    # --------------------------------------------------------

    method_box = FancyBboxPatch(
        (0.055, 0.070),
        0.89,
        0.085,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=SUMMARY_SOFT,
        edgecolor=SUMMARY_BORDER,
        linewidth=0.7,
        zorder=1
    )

    fig.patches.append(
        method_box
    )


    fig.text(
        0.070,
        0.137,
        "READING THE DASHBOARD",
        fontsize=6.7,
        fontweight="bold",
        color=SUMMARY_MUTED,
        ha="left",
        va="center"
    )


    fig.text(
        0.070,
        0.108,
        (
            "Positioning = current ratio vs trailing 3Y history   |   "
            "Momentum = annualized 63D log-ratio trend   |   "
            "Acceleration = change in momentum over 21 trading days"
        ),
        fontsize=6.75,
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.070,
        0.083,
        (
            "Green/red shading indicates relative direction or state, "
            "not an investment recommendation."
        ),
        fontsize=6.1,
        color=SUMMARY_MUTED,
        ha="left",
        va="center"
    )


    add_page_footer(
        fig,
        page_number
    )


    pdf.savefig(
        fig,
        facecolor=fig.get_facecolor()
    )

    plt.close(
        fig
    )

    page_number += 1


    # ========================================================
    # EM LOCAL BOND & CURRENCY MONITOR
    # ========================================================

    if EM_BOND_DECOMP is not None:

        fig = plt.figure(
            figsize=PAGE_SIZE,
            facecolor="white"
        )

        add_mcgill_banner(
            fig
        )


        # ====================================================
        # TITLE
        # ====================================================

        title_ax = make_text_panel(
            fig,
            [
                0.055,
                0.842,
                0.89,
                0.082
            ]
        )


        title_ax.text(
            0,
            0.78,
            "EM Local Bond & Currency Monitor",
            fontsize=18.5,
            fontweight="bold",
            va="center",
            transform=title_ax.transAxes
        )


        title_ax.text(
            0,
            0.34,
            (
                "LEMB CAD return attribution, duration-matched Canada "
                "comparison and matched EM FX signal"
            ),
            fontsize=8.2,
            alpha=0.72,
            transform=title_ax.transAxes
        )


        title_ax.text(
            0,
            0.01,
            (
                f"Weight source: {EM_BOND_DECOMP['weight_source']}  |  "
                f"Holdings as of: {EM_BOND_DECOMP['holdings_as_of']}  |  "
                f"Yahoo FX coverage: "
                f"{EM_BOND_DECOMP['fx_weight_coverage']:.1f}%  |  "
                f"Canada synthetic duration: "
                f"{EM_BOND_DECOMP['canada_synthetic_duration']:.2f}y"
            ),
            fontsize=6.65,
            alpha=0.66,
            transform=title_ax.transAxes
        )


        # ====================================================
        # MOST RECENT 1Y - RETURN ATTRIBUTION
        # ====================================================

        one_year_result = (
            EM_BOND_DECOMP[
                "period_attribution"
            ][
                "1Y"
            ]
        )


        # Large headline total-return figure.
        fig.text(
            0.055,
            0.806,
            "MOST RECENT 1 YEAR",
            fontsize=6.9,
            fontweight="bold",
            color=SUMMARY_MUTED,
            ha="left",
            va="center"
        )

        fig.text(
            0.055,
            0.775,
            "LEMB CAD Return",
            fontsize=9.0,
            fontweight="bold",
            color=SUMMARY_TEXT,
            ha="left",
            va="center"
        )

        fig.text(
            0.185,
            0.775,
            format_signed_percent(
                one_year_result[
                    "total"
                ]
            ),
            fontsize=12.5,
            fontweight="bold",
            color=SUMMARY_TEXT,
            ha="left",
            va="center"
        )


        # ----------------------------------------------------
        # LEFT: 1Y WATERFALL ATTRIBUTION
        # ----------------------------------------------------

        ax_waterfall = fig.add_axes([
            0.055,
            0.575,
            0.455,
            0.155
        ])


        waterfall_labels = [
            "Local Rates\n& Carry",
            "Matched EM FX",
            "Local x FX\nInteraction",
            "LEMB CAD\nTotal Return"
        ]


        waterfall_values = [
            one_year_result[
                "local"
            ],
            one_year_result[
                "fx"
            ],
            one_year_result[
                "interaction"
            ]
        ]


        running_total = 0.0
        waterfall_bottoms = []
        waterfall_heights = []


        for value in waterfall_values:

            waterfall_bottoms.append(
                running_total
            )

            waterfall_heights.append(
                value
            )

            running_total += value


        # Incremental components.
        ax_waterfall.bar(
            [0, 1, 2],
            waterfall_heights,
            bottom=waterfall_bottoms,
            width=0.62
        )


        # Total return bar.
        ax_waterfall.bar(
            [3],
            [
                one_year_result[
                    "total"
                ]
            ],
            width=0.62
        )


        # Connector lines between incremental bars.
        cumulative_levels = [
            waterfall_values[0],
            waterfall_values[0]
            + waterfall_values[1],
            waterfall_values[0]
            + waterfall_values[1]
            + waterfall_values[2]
        ]


        for connector_index in range(
            2
        ):

            ax_waterfall.plot(
                [
                    connector_index + 0.31,
                    connector_index + 1 - 0.31
                ],
                [
                    cumulative_levels[
                        connector_index
                    ],
                    cumulative_levels[
                        connector_index
                    ]
                ],
                linewidth=0.8,
                alpha=0.55
            )


        # Component labels.
        cumulative_for_labels = 0.0

        for idx, value in enumerate(
            waterfall_values
        ):

            cumulative_for_labels += value

            ax_waterfall.text(
                idx,
                cumulative_for_labels
                + (
                    0.20
                    if value >= 0
                    else -0.20
                ),
                f"{value:+.2f}%",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontsize=6.7,
                fontweight="bold"
            )


        ax_waterfall.text(
            3,
            one_year_result[
                "total"
            ]
            + (
                0.20
                if one_year_result[
                    "total"
                ] >= 0
                else -0.20
            ),
            f"{one_year_result['total']:+.2f}%",
            ha="center",
            va=(
                "bottom"
                if one_year_result[
                    "total"
                ] >= 0
                else "top"
            ),
            fontsize=6.9,
            fontweight="bold"
        )


        ax_waterfall.axhline(
            0,
            linewidth=0.8,
            alpha=0.75
        )


        # Explicit y-axis padding keeps value labels comfortably inside
        # the plotting area and away from the chart title.
        waterfall_levels = [
            0.0,
            waterfall_values[0],
            waterfall_values[0] + waterfall_values[1],
            waterfall_values[0] + waterfall_values[1] + waterfall_values[2],
            one_year_result["total"]
        ]

        waterfall_ymin = min(
            waterfall_levels
        )

        waterfall_ymax = max(
            waterfall_levels
        )

        waterfall_span = max(
            waterfall_ymax - waterfall_ymin,
            1.0
        )

        ax_waterfall.set_ylim(
            waterfall_ymin - 0.12 * waterfall_span,
            waterfall_ymax + 0.30 * waterfall_span
        )


        ax_waterfall.set_xticks(
            [
                0,
                1,
                2,
                3
            ]
        )

        ax_waterfall.set_xticklabels(
            waterfall_labels
        )


        ax_waterfall.set_title(
            "1-Year LEMB CAD Return Attribution",
            fontsize=8.7,
            fontweight="bold",
            pad=8
        )


        ax_waterfall.set_ylabel(
            "Return (%)",
            fontsize=6.8,
            labelpad=5
        )


        ax_waterfall.tick_params(
            axis="x",
            labelsize=5.8,
            pad=4
        )

        ax_waterfall.tick_params(
            axis="y",
            labelsize=6.0,
            pad=2
        )


        ax_waterfall.grid(
            axis="y",
            alpha=0.16
        )


        # ----------------------------------------------------
        # RIGHT: 1Y RELATIVE BOND PERFORMANCE
        # ----------------------------------------------------

        ax_relative = fig.add_axes([
            0.595,
            0.575,
            0.350,
            0.155
        ])


        relative_labels = [
            "EM Local Rates\n& Carry",
            "Canada Federal\n(Duration-Matched)",
            "EM Local Minus\nCanada"
        ]


        relative_values = [
            one_year_result[
                "local"
            ],
            one_year_result[
                "canada"
            ],
            one_year_result[
                "local_vs_canada"
            ]
        ]


        ax_relative.bar(
            relative_labels,
            relative_values,
            width=0.60
        )


        ax_relative.axhline(
            0,
            linewidth=0.8,
            alpha=0.75
        )


        # Leave enough headroom / footroom for value labels.
        relative_ymin = min(
            [0.0] + relative_values
        )

        relative_ymax = max(
            [0.0] + relative_values
        )

        relative_span = max(
            relative_ymax - relative_ymin,
            1.0
        )

        ax_relative.set_ylim(
            relative_ymin - 0.20 * relative_span,
            relative_ymax + 0.30 * relative_span
        )


        for idx, value in enumerate(
            relative_values
        ):

            ax_relative.text(
                idx,
                value
                + (
                    0.18
                    if value >= 0
                    else -0.18
                ),
                f"{value:+.2f}%",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontsize=6.7,
                fontweight="bold"
            )


        ax_relative.set_title(
            "1-Year EM Local Bond Return vs\n"
            "Duration-Matched Canadian Federal Bonds",
            fontsize=8.0,
            fontweight="bold",
            pad=7,
            linespacing=1.08
        )


        ax_relative.set_ylabel(
            "1Y Return (%)",
            fontsize=6.8,
            labelpad=6
        )


        ax_relative.tick_params(
            axis="x",
            labelsize=5.6,
            pad=4
        )

        ax_relative.tick_params(
            axis="y",
            labelsize=6.0,
            pad=2
        )


        ax_relative.grid(
            axis="y",
            alpha=0.16
        )


        # ====================================================
        # YTD / 1Y / 3Y SUMMARY - SPLIT INTO TWO QUESTIONS
        # ====================================================

        decomposition_rows = []
        relative_rows = []


        for period_label in [
            "YTD",
            "1Y",
            "3Y"
        ]:

            period_result = (
                EM_BOND_DECOMP[
                    "period_attribution"
                ][
                    period_label
                ]
            )


            decomposition_rows.append([
                period_label,

                format_signed_percent(
                    period_result[
                        "total"
                    ]
                ),

                format_signed_percent(
                    period_result[
                        "local"
                    ]
                ),

                format_signed_percent(
                    period_result[
                        "fx"
                    ]
                ),

                format_signed_percent(
                    period_result[
                        "interaction"
                    ]
                )
            ])


            relative_rows.append([
                period_label,

                format_signed_percent(
                    period_result[
                        "local"
                    ]
                ),

                format_signed_percent(
                    period_result[
                        "canada"
                    ]
                ),

                format_signed_percent(
                    period_result[
                        "local_vs_canada"
                    ]
                )
            ])


        # ----------------------------------------------------
        # LEFT TABLE: RETURN DECOMPOSITION
        # ----------------------------------------------------

        fig.text(
            0.055,
            0.490,
            "LEMB CAD Return Decomposition",
            fontsize=9.4,
            fontweight="bold",
            color=SUMMARY_TEXT,
            ha="left",
            va="center"
        )


        table_decomp_ax = fig.add_axes([
            0.055,
            0.365,
            0.89,
            0.105
        ])

        table_decomp_ax.axis(
            "off"
        )


        decomposition_table = (
            table_decomp_ax.table(

                cellText=
                    decomposition_rows,

                colLabels=[
                    "Period",
                    "LEMB Total Return\n(CAD)",
                    "Local Rates & Carry\nReturn",
                    "Matched EM FX\nReturn vs CAD",
                    "Local x FX\nInteraction"
                ],

                colWidths=[
                    0.10,
                    0.20,
                    0.25,
                    0.23,
                    0.22
                ],

                bbox=[
                    0,
                    0,
                    1,
                    1
                ],

                cellLoc="center",
                colLoc="center"
            )
        )


        style_table(
            decomposition_table,
            body_fontsize=7.2
        )

        decomposition_table.scale(
            1.0,
            1.18
        )

        for cell in decomposition_table.get_celld().values():
            cell.PAD = 0.045

        for col in range(5):
            decomposition_table[
                (0, col)
            ].get_text().set_fontsize(
                6.9
            )

            decomposition_table[
                (0, col)
            ].set_height(
                decomposition_table[
                    (0, col)
                ].get_height()
                * 1.45
            )


        # ----------------------------------------------------
        # RIGHT TABLE: RELATIVE BOND PERFORMANCE
        # ----------------------------------------------------

        fig.text(
            0.055,
            0.330,
            "EM Local Bonds vs Duration-Matched Canada — Relative Performance",
            fontsize=9.4,
            fontweight="bold",
            color=SUMMARY_TEXT,
            ha="left",
            va="center"
        )


        table_relative_ax = fig.add_axes([
            0.055,
            0.205,
            0.89,
            0.105
        ])

        table_relative_ax.axis(
            "off"
        )


        relative_table = (
            table_relative_ax.table(

                cellText=
                    relative_rows,

                colLabels=[
                    "Period",
                    "EM Local Rates & Carry\nReturn",
                    "Duration-Matched Canadian Federal Bond\nReturn",
                    "EM Local vs Canada\nExcess Return"
                ],

                colWidths=[
                    0.10,
                    0.29,
                    0.34,
                    0.27
                ],

                bbox=[
                    0,
                    0,
                    1,
                    1
                ],

                cellLoc="center",
                colLoc="center"
            )
        )


        style_table(
            relative_table,
            body_fontsize=7.2
        )

        relative_table.scale(
            1.0,
            1.18
        )

        for cell in relative_table.get_celld().values():
            cell.PAD = 0.045

        for col in range(4):
            relative_table[
                (0, col)
            ].get_text().set_fontsize(
                6.9
            )

            relative_table[
                (0, col)
            ].set_height(
                relative_table[
                    (0, col)
                ].get_height()
                * 1.45
            )


        # ====================================================
        # CURRENT SIGNALS
        # ====================================================

        em_local_result = next(
            (
                result
                for result in results
                if result[
                    "ratio_name"
                ] == "EM Local / Synthetic Canada"
            ),
            None
        )


        em_fx_result = next(
            (
                result
                for result in results
                if result[
                    "ratio_name"
                ] == "LEMB-Matched EM FX / CAD"
            ),
            None
        )


        fig.text(
            0.055,
            0.165,
            "Current Structural Signals",
            fontsize=9.4,
            fontweight="bold",
            color=SUMMARY_TEXT,
            ha="left",
            va="center"
        )


        def draw_em_signal_card(
            fig,
            x,
            y,
            w,
            h,
            result,
            title
        ):

            card = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor=SUMMARY_SOFT,
                edgecolor=SUMMARY_BORDER,
                linewidth=0.7,
                zorder=1
            )

            fig.patches.append(
                card
            )


            if result is None:

                fig.text(
                    x + 0.014,
                    y + h * 0.58,
                    title,
                    fontsize=8.3,
                    fontweight="bold",
                    color=SUMMARY_TEXT,
                    ha="left",
                    va="center"
                )

                fig.text(
                    x + 0.014,
                    y + h * 0.30,
                    "Signal unavailable",
                    fontsize=7.0,
                    color=SUMMARY_MUTED,
                    ha="left",
                    va="center"
                )

                return


            fig.text(
                x + 0.014,
                y + h * 0.76,
                title,
                fontsize=8.2,
                fontweight="bold",
                color=SUMMARY_TEXT,
                ha="left",
                va="center"
            )


            fig.text(
                x + 0.014,
                y + h * 0.46,
                (
                    f"3Y Z {result['zscore_3y']:+.2f}   |   "
                    f"Percentile {result['percentile_3y']:.1f}%   |   "
                    f"Positioning {result['positioning']}"
                ),
                fontsize=6.4,
                color=SUMMARY_TEXT,
                ha="left",
                va="center"
            )


            fig.text(
                x + 0.014,
                y + h * 0.17,
                (
                    f"Momentum: {result['momentum']}   |   "
                    f"Acceleration: {result['acceleration']}"
                ),
                fontsize=6.4,
                fontweight="bold",
                color=SUMMARY_TEXT,
                ha="left",
                va="center"
            )


        draw_em_signal_card(
            fig,
            0.055,
            0.090,
            0.425,
            0.060,
            em_local_result,
            "EM Local / Synthetic Canada"
        )


        draw_em_signal_card(
            fig,
            0.520,
            0.090,
            0.425,
            0.060,
            em_fx_result,
            "LEMB-Matched EM FX vs CAD"
        )


        # ====================================================
        # METHODOLOGY / WEIGHT NOTE
        # ====================================================

        top_weights = (
            EM_BOND_DECOMP[
                "used_currency_weights"
            ]
            .head(6)
        )


        top_weight_text = " | ".join(
            [
                f"{currency} {weight:.1f}%"

                for currency, weight
                in top_weights.items()
            ]
        )


        fig.text(
            0.055,
            0.054,
            (
                "Matched FX largest weights: "
                + top_weight_text
            ),
            fontsize=5.5,
            alpha=0.72,
            ha="left",
            va="bottom"
        )


        fig.text(
            0.055,
            0.070,
            (
                "Signal interpretation: EM Local / Synthetic Canada rising = "
                "EM local rates & carry outperforming duration-matched Canada; "
                "LEMB-Matched EM FX vs CAD rising = LEMB-weighted EM currencies "
                "strengthening versus CAD."
            ),
            fontsize=5.6,
            alpha=0.72,
            ha="left",
            va="bottom"
        )


        fig.text(
            0.055,
            0.036,
            (
                "Methodology: LEMB CAD = adjusted LEMB USD x USD/CAD. "
                "Local Rates & Carry = LEMB CAD / matched FX basket. "
                f"Canada = {EM_BOND_DECOMP['canada_zfs_weight']:.1%} ZFS + "
                f"{EM_BOND_DECOMP['canada_zfm_weight']:.1%} ZFM, targeting "
                f"{EM_BOND_DECOMP['lemb_target_duration']:.2f}y duration. "
                "Current LEMB currency weights are applied historically."
            ),
            fontsize=5.5,
            alpha=0.68,
            ha="left",
            va="bottom",
            wrap=True
        )


        add_page_footer(
            fig,
            page_number
        )


        pdf.savefig(
            fig,
            facecolor=fig.get_facecolor()
        )


        plt.close(
            fig
        )


        page_number += 1


    # ========================================================
    # INDIVIDUAL INDICATOR PAGES
    # ========================================================

    for result in results:

        ratio_name = (
            result["ratio_name"]
        )

        # Record the final 1-based PDF page number before this
        # indicator page is created. This remains correct even if
        # overview pages are added, removed or reordered.
        indicator_page_numbers[
            ratio_name
        ] = page_number

        series = (
            result["series"]
        )


        data_3y = trailing_data(
            series,
            3
        )


        data_1y = trailing_data(
            series,
            1
        )


        if (
            data_3y.empty
            or data_1y.empty
        ):
            continue


        mean_3y = (
            data_3y.mean()
        )

        std_3y = (
            data_3y.std()
        )


        upper_1 = (
            mean_3y
            + std_3y
        )

        lower_1 = (
            mean_3y
            - std_3y
        )

        upper_2 = (
            mean_3y
            + 2 * std_3y
        )

        lower_2 = (
            mean_3y
            - 2 * std_3y
        )


        # ----------------------------------------------------
        # FITTED TREND LINES FOR 1Y CHART
        # ----------------------------------------------------

        (
            current_trend_dates,
            current_trend_values,
            current_trend_slope
        ) = fitted_log_trend_line(
            series,
            window=MOMENTUM_WINDOW,
            end_offset=0
        )


        (
            previous_trend_dates,
            previous_trend_values,
            previous_trend_slope
        ) = fitted_log_trend_line(
            series,
            window=MOMENTUM_WINDOW,
            end_offset=ACCELERATION_LOOKBACK
        )


        fig = plt.figure(
            figsize=PAGE_SIZE,
            facecolor="white"
        )


        add_mcgill_banner(
            fig
        )


        # ====================================================
        # HEADER
        # ====================================================

        header_ax = make_text_panel(
            fig,
            [
                0.055,
                0.842,
                0.89,
                0.078
            ]
        )


        header_ax.text(
            0,
            0.92,
            result["bucket"],
            fontsize=7.7,
            alpha=0.70,
            va="top",
            transform=header_ax.transAxes
        )


        header_ax.text(
            0,
            0.50,
            result["short_name"],
            fontsize=18,
            fontweight="bold",
            va="center",
            transform=header_ax.transAxes
        )


        header_ax.text(
            0,
            0.02,
            ratio_name,
            fontsize=9.1,
            va="bottom",
            transform=header_ax.transAxes
        )


        # ====================================================
        # COMPACT SIGNAL STRIP
        # ====================================================

        signal_ax = make_text_panel(
            fig,
            [
                0.055,
                0.752,
                0.89,
                0.062
            ]
        )


        signal_ax.text(
            0.00,
            0.93,
            "Purpose",
            fontsize=8.2,
            fontweight="bold",
            va="top",
            transform=signal_ax.transAxes
        )


        signal_ax.text(
            0.00,
            0.48,
            fill(
                result["purpose"],
                width=78
            ),
            fontsize=7.1,
            va="top",
            linespacing=1.10,
            transform=signal_ax.transAxes
        )


        signal_items = [
            (
                0.62,
                "Positioning",
                result["positioning"]
            ),
            (
                0.755,
                "Momentum",
                result["momentum"]
            ),
            (
                0.875,
                "Acceleration",
                result["acceleration"]
            )
        ]


        for (
            x_position,
            label,
            value
        ) in signal_items:

            signal_ax.text(
                x_position,
                0.93,
                label,
                fontsize=7.8,
                fontweight="bold",
                va="top",
                transform=signal_ax.transAxes
            )

            signal_ax.text(
                x_position,
                0.47,
                value,
                fontsize=7.2,
                fontweight="bold",
                va="top",
                transform=signal_ax.transAxes
            )


        # ====================================================
        # COMPACT SMART COMMENT
        # ====================================================

        comment_ax = make_text_panel(
            fig,
            [
                0.055,
                0.665,
                0.89,
                0.070
            ]
        )


        comment_ax.text(
            0,
            0.96,
            (
                "Comment - "
                + result[
                    "smart_headline"
                ]
            ),
            fontsize=8.8,
            fontweight="bold",
            va="top",
            transform=comment_ax.transAxes
        )


        comment_ax.text(
            0,
            0.58,
            fill(
                result[
                    "smart_comment"
                ],
                width=178
            ),
            fontsize=6.9,
            va="top",
            linespacing=1.12,
            transform=comment_ax.transAxes
        )


        # ====================================================
        # LEFT: COMPACT STATISTICS
        # ====================================================

        left_stats_ax = make_text_panel(
            fig,
            [
                0.055,
                0.390,
                0.455,
                0.235
            ]
        )


        left_stats_ax.text(
            0.00,
            0.98,
            "Historical Statistics",
            fontsize=8.7,
            fontweight="bold",
            va="top",
            transform=left_stats_ax.transAxes
        )


        left_stats_ax.text(
            0.54,
            0.98,
            "Positioning & Trend",
            fontsize=8.7,
            fontweight="bold",
            va="top",
            transform=left_stats_ax.transAxes
        )


        current_date_text = (

            str(
                result[
                    "current_date"
                ].date()
            )

            if result[
                "current_date"
            ] is not None

            else "N/A"
        )


        historical_rows = [
            (
                "Available history",
                f"{result['history_years']:.1f} years"
            ),
            (
                "10Y / available avg",
                format_number(
                    result["avg_10y"]
                )
            ),
            (
                "5Y average",
                format_number(
                    result["avg_5y"]
                )
            ),
            (
                "3Y average",
                format_number(
                    result["avg_3y"]
                )
            ),
            (
                "1Y average",
                format_number(
                    result["avg_1y"]
                )
            ),
            (
                "Latest ratio",
                (
                    f"{format_number(result['current_ratio'])} "
                    f"({current_date_text})"
                )
            )
        ]


        trend_rows = [
            (
                "vs 1Y avg",
                format_signed_percent(
                    result[
                        "deviation_from_1y"
                    ]
                )
            ),
            (
                "vs 3Y avg",
                format_signed_percent(
                    result[
                        "deviation_from_3y"
                    ]
                )
            ),
            (
                "3Y z-score",
                format_signed(
                    result[
                        "zscore_3y"
                    ]
                )
            ),
            (
                "3Y percentile",
                format_percent(
                    result[
                        "percentile_3y"
                    ]
                )
            ),
            (
                "63D momentum",
                format_signed_percent(
                    result[
                        "momentum_value"
                    ]
                )
            ),
            (
                "Momentum t-stat",
                format_signed(
                    result[
                        "momentum_tstat"
                    ]
                )
            ),
            (
                "21D acceleration",
                format_signed_percent(
                    result[
                        "acceleration_value"
                    ]
                )
            ),
            (
                "Acceleration 3Y z",
                format_signed(
                    result[
                        "acceleration_z"
                    ]
                )
            )
        ]


        y_hist = 0.82

        for label, value in historical_rows:

            left_stats_ax.text(
                0.00,
                y_hist,
                label,
                fontsize=6.9,
                va="center",
                transform=left_stats_ax.transAxes
            )

            left_stats_ax.text(
                0.47,
                y_hist,
                value,
                fontsize=6.9,
                ha="right",
                va="center",
                transform=left_stats_ax.transAxes
            )

            y_hist -= 0.125


        y_trend = 0.82

        for label, value in trend_rows:

            left_stats_ax.text(
                0.54,
                y_trend,
                label,
                fontsize=6.7,
                va="center",
                transform=left_stats_ax.transAxes
            )

            left_stats_ax.text(
                0.99,
                y_trend,
                value,
                fontsize=6.9,
                fontweight="bold",
                ha="right",
                va="center",
                transform=left_stats_ax.transAxes
            )

            y_trend -= 0.095


        # ====================================================
        # BOTTOM-LEFT: 3Y POSITIONING CHART
        # ====================================================

        ax_3y = fig.add_axes([
            0.070,
            0.095,
            0.435,
            0.245
        ])


        ax_3y.plot(
            data_3y.index,
            data_3y.values,
            linewidth=1.35,
            label="Ratio"
        )


        ax_3y.axhline(
            mean_3y,
            linestyle="--",
            linewidth=1.0,
            label="3Y Mean"
        )


        ax_3y.fill_between(
            data_3y.index,
            lower_1,
            upper_1,
            alpha=0.10,
            label="+/- 1 SD"
        )


        ax_3y.axhline(
            upper_2,
            linestyle=":",
            linewidth=0.85,
            label="+2 SD"
        )


        ax_3y.axhline(
            lower_2,
            linestyle=":",
            linewidth=0.85,
            label="-2 SD"
        )


        if not pd.isna(
            result[
                "current_ratio"
            ]
        ):

            ax_3y.scatter(
                data_3y.index[-1],
                result[
                    "current_ratio"
                ],
                s=38,
                zorder=5,
                label="Current"
            )


        ax_3y.set_title(
            "Trailing 3-Year Historical Positioning",
            pad=5,
            fontsize=8.9,
            fontweight="bold"
        )


        ax_3y.set_ylabel(
            "Relative Ratio",
            fontsize=7.4
        )


        ax_3y.tick_params(
            axis="both",
            labelsize=6.7
        )


        ax_3y.margins(
            x=0.01
        )


        ax_3y.grid(
            alpha=0.20
        )


        ax_3y.legend(
            loc="upper left",
            fontsize=5.8,
            ncol=3,
            frameon=True
        )


        # ====================================================
        # RIGHT: ENLARGED SQUARE 1Y TREND CHART
        # ====================================================

        ax_1y = fig.add_axes([
            0.565,
            0.105,
            0.365,
            0.472
        ])


        ax_1y.plot(
            data_1y.index,
            data_1y.values,
            linewidth=1.65,
            label="Ratio"
        )


        # Current 63D fitted trend - this is the trend used
        # for the current momentum calculation.
        if len(
            current_trend_dates
        ) > 0:

            ax_1y.plot(
                current_trend_dates,
                current_trend_values,
                linewidth=2.0,
                label=(
                    "Current 63D trend "
                    f"({current_trend_slope:+.1f}% ann.)"
                )
            )


        # Previous fitted trend from 21 trading days ago.
        # Comparing its slope with the current line visually
        # represents momentum acceleration / deceleration.
        if len(
            previous_trend_dates
        ) > 0:

            ax_1y.plot(
                previous_trend_dates,
                previous_trend_values,
                linestyle="--",
                linewidth=1.55,
                label=(
                    "63D trend 21D ago "
                    f"({previous_trend_slope:+.1f}% ann.)"
                )
            )


        if not pd.isna(
            result[
                "current_ratio"
            ]
        ):

            ax_1y.scatter(
                data_1y.index[-1],
                result[
                    "current_ratio"
                ],
                s=48,
                zorder=6,
                label="Current"
            )


        ax_1y.set_title(
            (
                "Most Recent 1 Year\n"
                "Momentum & Acceleration"
            ),
            pad=7,
            fontsize=10.0,
            fontweight="bold"
        )


        ax_1y.set_ylabel(
            "Relative Ratio",
            fontsize=8.0
        )


        ax_1y.tick_params(
            axis="both",
            labelsize=7.0
        )


        ax_1y.margins(
            x=0.015
        )


        ax_1y.grid(
            alpha=0.20
        )


        ax_1y.legend(
            loc="best",
            fontsize=6.2,
            frameon=True
        )


        # Compact explanatory annotation.
        ax_1y.text(
            0.02,
            0.02,
            (
                f"Momentum: {result['momentum']}  |  "
                f"Acceleration: {result['acceleration']}"
            ),
            fontsize=6.4,
            fontweight="bold",
            transform=ax_1y.transAxes,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#D9DDE3",
                alpha=0.90
            )
        )


        add_page_footer(
            fig,
            page_number
        )


        pdf.savefig(
            fig,
            facecolor=fig.get_facecolor()
        )

        plt.close(
            fig
        )

        page_number += 1


    # ========================================================
    # FINAL PAGE - CALCULATION CONVENTIONS & METHODOLOGY
    # ========================================================

    fig = plt.figure(
        figsize=PAGE_SIZE,
        facecolor="white"
    )

    add_mcgill_banner(
        fig
    )


    # ========================================================
    # TITLE
    # ========================================================

    title_ax = make_text_panel(
        fig,
        [
            0.055,
            0.842,
            0.89,
            0.085
        ]
    )

    title_ax.text(
        0,
        0.72,
        "Calculation Conventions & Methodology",
        fontsize=19,
        fontweight="bold",
        va="center",
        transform=title_ax.transAxes
    )

    title_ax.text(
        0,
        0.12,
        (
            "Summary of data treatment, statistical measures "
            "and classification rules used throughout the report"
        ),
        fontsize=8.4,
        alpha=0.75,
        transform=title_ax.transAxes
    )


    # ========================================================
    # SECTION DRAWING HELPER
    # ========================================================

    def draw_method_section(
        ax,
        y,
        title,
        body,
        body_width=69,
        title_size=9.7,
        body_size=6.55,
        line_spacing=1.16
    ):

        ax.text(
            0,
            y,
            title,
            fontsize=title_size,
            fontweight="bold",
            va="top",
            transform=ax.transAxes
        )

        ax.text(
            0,
            y - 0.055,
            fill(
                body,
                width=body_width
            ),
            fontsize=body_size,
            va="top",
            linespacing=line_spacing,
            transform=ax.transAxes
        )


    # ========================================================
    # MAIN TWO-COLUMN CONTENT
    # ========================================================

    left_ax = make_text_panel(
        fig,
        [
            0.055,
            0.300,
            0.405,
            0.500
        ]
    )

    right_ax = make_text_panel(
        fig,
        [
            0.535,
            0.300,
            0.410,
            0.500
        ]
    )


    draw_method_section(
        left_ax,
        0.98,
        "1. Market Data",
        (
            "Daily market data are downloaded from Yahoo Finance using "
            "yfinance. The script requests Close with auto_adjust=True, "
            "so the series are adjusted for corporate actions such as "
            "splits and distributions. They are adjusted-price / "
            "total-return-like series rather than raw closes."
        )
    )


    draw_method_section(
        left_ax,
        0.785,
        "2. Relative-Performance Ratios",
        (
            "For most indicators: Relative Ratio = Numerator Adjusted Price / "
            "Denominator Adjusted Price. A rising ratio means the numerator is "
            "outperforming the denominator; a falling ratio means it is "
            "underperforming."
        )
    )


    draw_method_section(
        left_ax,
        0.605,
        "3. Resources vs World",
        (
            "IXC and MXI are converted to daily percentage returns. A 50% IXC "
            "+ 50% MXI daily-rebalanced return is compounded into a synthetic "
            "resource index. URTH is normalized over the same dates. "
            "Resources vs World = Synthetic Resource Index / Normalized URTH."
        )
    )


    draw_method_section(
        left_ax,
        0.390,
        "4. Historical Averages",
        (
            "1Y, 3Y, 5Y and 10Y averages are arithmetic means of available "
            "daily ratio observations within each trailing calendar-year "
            "window. If less than 10 years of history exists, "
            "'10Y / available avg' uses the observations actually available."
        )
    )


    draw_method_section(
        left_ax,
        0.195,
        "5. Positioning Statistics",
        (
            "Deviation (%) = (Current Ratio / Historical Average - 1) x 100. "
            "The 3Y z-score measures the current ratio's distance from its "
            "trailing 3-year mean in standard deviations. The 3Y percentile "
            "is the share of trailing 3-year daily observations less than or "
            "equal to the current ratio."
        )
    )


    draw_method_section(
        right_ax,
        0.98,
        "6. Momentum",
        (
            "Momentum is estimated from a 63-trading-day ordinary least-squares "
            "regression of log(Relative Ratio) on time. The fitted daily slope "
            "is multiplied by 252 and by 100 to express the trend as an "
            "approximate annualized relative-performance rate."
        )
    )


    draw_method_section(
        right_ax,
        0.775,
        "7. Momentum Strength",
        (
            "The regression slope t-statistic distinguishes persistent trend "
            "from noise. Strong Positive requires a positive slope with "
            "t-stat >= +2; Positive requires t-stat >= +1. Strong Negative "
            "requires a negative slope with t-stat <= -2; Negative requires "
            "t-stat <= -1. Other readings are Flat / Noisy."
        )
    )


    draw_method_section(
        right_ax,
        0.535,
        "8. Momentum Acceleration",
        (
            "Acceleration = Current 63D Annualized Momentum - 63D Annualized "
            "Momentum 21 trading days earlier. This finite difference "
            "approximates the second derivative of the log relative-performance "
            "path: positive values mean the trend is becoming more positive; "
            "negative values mean the trend is becoming more negative."
        )
    )


    draw_method_section(
        right_ax,
        0.280,
        "9. Acceleration Strength",
        (
            "Acceleration is standardized against its own trailing 3-year "
            "history. A positive acceleration with z-score >= +1 is Strongly "
            "Accelerating; any other positive reading is Accelerating. A "
            "negative acceleration with z-score <= -1 is Strongly Decelerating; "
            "any other negative reading is Decelerating."
        )
    )


    # ========================================================
    # FULL-WIDTH INTERPRETATION SECTION
    # ========================================================

    fig.text(
        0.055,
        0.245,
        "10. Interpretation",
        fontsize=9.7,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="top"
    )

    fig.text(
        0.055,
        0.213,
        fill(
            (
                "Positioning describes where the ratio is historically; "
                "Momentum describes the current direction and persistence of "
                "relative performance; Acceleration describes whether that "
                "momentum is strengthening or weakening. The EM local-bond "
                "decomposition uses LEMB's latest reported market-currency "
                "weights to build a custom CAD currency basket from Yahoo FX "
                "rates. The Canadian comparison is a synthetic ZFS/ZFM federal "
                "bond benchmark calibrated to LEMB's target duration. The "
                "resulting EM Local / Synthetic Canada relative index and "
                "LEMB-matched EM FX vs CAD index are then passed through the "
                "same positioning, momentum and acceleration framework."
            ),
            width=158
        ),
        fontsize=6.45,
        color=SUMMARY_TEXT,
        ha="left",
        va="top",
        linespacing=1.16
    )


    # ========================================================
    # BOTTOM CLASSIFICATION STRIP
    # ========================================================

    bottom_box = FancyBboxPatch(
        (0.055, 0.065),
        0.89,
        0.070,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=SUMMARY_BORDER,
        linewidth=0.7,
        zorder=1
    )

    fig.patches.append(
        bottom_box
    )


    fig.text(
        0.070,
        0.116,
        "Positioning",
        fontsize=7.0,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )

    fig.text(
        0.155,
        0.116,
        (
            "Extremely High: Z >= +2   |   High: +1 to +2   |   "
            "Neutral: -1 to +1   |   Low: -2 to -1   |   "
            "Extremely Low: Z <= -2"
        ),
        fontsize=6.2,
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    fig.text(
        0.070,
        0.084,
        "Trend windows",
        fontsize=7.0,
        fontweight="bold",
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )

    fig.text(
        0.155,
        0.084,
        (
            "Momentum = 63 trading days (~3 months)   |   "
            "Acceleration lookback = 21 trading days (~1 month)   |   "
            "Annualization = 252 trading days"
        ),
        fontsize=6.2,
        color=SUMMARY_TEXT,
        ha="left",
        va="center"
    )


    add_page_footer(
        fig,
        page_number
    )


    pdf.savefig(
        fig,
        facecolor=fig.get_facecolor()
    )

    plt.close(
        fig
    )

    page_number += 1


# ============================================================
# ADD INTERNAL PAGE-1 HYPERLINKS
# ============================================================

add_internal_indicator_links(
    OUTPUT_PDF,
    summary_link_rectangles,
    indicator_page_numbers
)


# ============================================================
# FINISHED
# ============================================================

print()
print(
    "Market sentiment PDF report completed successfully."
)

print()
print(
    "Report saved to:"
)

print(
    OUTPUT_PDF
)

print()
print(
    f"Total PDF pages: {page_number - 1}"
)

print()
print(
    "Includes final Calculation Conventions & Methodology page"
)