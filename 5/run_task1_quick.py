import csv
import math
from pathlib import Path


def read_xy(path):
    xs = []
    ys = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            xs.append(x)
            ys.append(y)
    return xs, ys


def fit_ols(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        slope = 0.0
    else:
        slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def mse(xs, ys, slope, intercept):
    n = len(xs)
    se = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return se / n


def analyze(path):
    p = Path(path)
    if not p.exists():
        print(f"Missing: {path}")
        return
    xs, ys = read_xy(path)
    slope, intercept = fit_ols(xs, ys)
    err = mse(xs, ys, slope, intercept)
    print(f"Dataset: {path}")
    print(f"  samples: {len(xs)}")
    print(f"  slope: {slope:.6f}, intercept: {intercept:.6f}")
    print(f"  MSE (OLS): {err:.6f}")
    # heuristic: if MSE is small relative to variance of y, linear model is appropriate
    var_y = sum((y - sum(ys)/len(ys))**2 for y in ys) / len(ys)
    if var_y == 0:
        print("  variance(y)=0 — trivial target\n")
        return
    rel = err / var_y
    print(f"  variance(y): {var_y:.6f}, relative MSE: {rel:.6f}")
    if rel < 0.01:
        print("  Conclusion: linear model fits very well (learnable by LinearRegressionNN)\n")
    elif rel < 0.1:
        print("  Conclusion: linear model fits reasonably well (LinearRegressionNN appropriate)\n")
    else:
        print("  Conclusion: linear model does not fit well — consider more complex model\n")


def main():
    paths = [
        "../Lab4-Regression-using-PyTorch/linear-regression-data1.csv",
        "../Lab4-Regression-using-PyTorch/assignment-data.csv",
        "../Lab4-Regression-using-PyTorch/assignment-data2.csv",
    ]
    for p in paths:
        analyze(p)


if __name__ == '__main__':
    main()
