import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.signal import savgol_filter

linestyles = ['-r', '--', '-.', ':']

def extract_values(log_files, keyword):
    values = []
    names = []

    for log_file in log_files:
        with open(log_file, 'r') as file:
            contents = file.read()
            matches = re.findall(f'{keyword} (\d+.\d+)', contents)
            floats = [float(x) for x in matches]

            names.extend(re.findall('_\d+_\d+_', log_file))
            if matches:
                values.append(floats)

    return values, names


def exponential_function(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_curve(x, y):
    p0 = [-1000, -0.2, 32]
    popt, _ = opt.curve_fit(exponential_function, x, y, p0=p0)
    return popt

def create_plot(x, y, i, name):
    xdata = np.arange(100, (len(y) + 1) * 100, 100)
    xnew = np.linspace(min(xdata), max(xdata), len(xdata) * 10)
    labels = name[1:-1].split('_')
    label = labels[0] + " width, " + labels[1] + " depth"
    yhat = savgol_filter(y, 51, 3)
    plt.plot(xdata, yhat, linestyles[i], label=label)

    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title('Ablation Study of PSNR')
    plt.legend()

def main():
    parser = argparse.ArgumentParser(description='Log Analysis')
    parser.add_argument('log_files', metavar='log_file', nargs='+', help='path(s) to the log file(s)')
    parser.add_argument('--keyword', required=True, help='keyword to search for')

    args = parser.parse_args()

    values, names = extract_values(args.log_files, args.keyword)
    if values:
        for i, v in enumerate(values):
            x = np.arange(1, len(v) + 1)
            create_plot(x, v, i, names[i])
        plt.show()
    else:
        print('No values found.')

if __name__ == '__main__':
    main()
