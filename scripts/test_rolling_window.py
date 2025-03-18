import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normalize


def squared_kernel(size):
    kernel = np.linspace(0, 1, N) * (1.0 - np.linspace(0, 1, N))
    scaling_factor = 6.0 * (N - 1.0) / (N * (N - 2))
    return kernel * scaling_factor


# apply a linear weighted sum of window, preserves integral
def own_flattening_function(values):
    N = len(values)
    if N <= 2:
        return 0.0
    kernel = gaussian_kernel(N, 0.1)
    return (values * kernel).sum()


def custom_running_mean(data: pd.DataFrame, n_days: int) -> pd.DataFrame:
    # step 1: fill daily indexes
    data = data.resample("D").sum()  # .fillna(value=0.0)
    first_date = data.index.iloc[0]
    print(first_date)

    window = data.rolling(
        window=datetime.timedelta(days=days_running_mean),
    )
    result = window.apply(own_flattening_function, raw=True)
    return result


n_data = 100
days_running_mean = 20


data = pd.Series(
    np.concatenate(
        (
            np.zeros(days_running_mean),
            abs(np.random.randn(n_data)),
            np.zeros(days_running_mean),
        ),
        axis=0,
    ),
    index=pd.date_range("2025-01-01", periods=n_data + 2 * days_running_mean),
)


print(data)

for i in range(20):
    data = data.drop(data.index[50])


rolling_window = custom_running_mean(data, days_running_mean)
# check integral
print(data.sum())
print(rolling_window.sum())

plt.plot(data)
plt.plot(rolling_window)


plt.savefig("out.pdf")
