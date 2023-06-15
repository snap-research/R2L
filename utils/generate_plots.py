import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.signal import savgol_filter

linestyles = ['-r', '--', '-.', ':']

def extract_values(log_files, keyword):
    values = []

    for log_file in log_files:
        with open(log_file, 'r') as file:
            contents = file.read()
            matches = re.findall(f'{keyword} (\d+.\d+)', contents)
            floats = [float(x) for x in matches]

            if matches:
                values.append(floats)

    return values


def exponential_function(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_curve(x, y):
    p0 = [-1000, -0.2, 32]
    popt, _ = opt.curve_fit(exponential_function, x, y, p0=p0)
    return popt

def create_plot(x, y, i, name):
    label = name
    plt.plot(x, y, linestyles[i], label=label)

    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    # plt.title('Ablation Study of PSNR')
    plt.legend()

def main():
    parser = argparse.ArgumentParser(description='Log Analysis')
    parser.add_argument('log_files', metavar='log_file', nargs='+', help='path(s) to the log file(s)')
    parser.add_argument('--keyword', required=True, help='keyword to search for')

    args = parser.parse_args()

    train_values = extract_values(args.log_files, args.keyword)
    test_values = extract_values(args.log_files, ' TestPSNR')
    train_names = ["128W88D train", "128W44D train", "88W44D train", "64W32D train"]
    test_names = ["128W88D test", "128W44D test", "88W44D test", "64W32D test"]
    if train_values:
        for i, v in enumerate(train_values):
            print(max(v))
            xdata = np.arange(100, (len(v) + 1) * 100, 100)
            yhat = savgol_filter(v, 51, 3)
            create_plot(xdata, yhat, i, train_names[i])
        for i, v in enumerate(test_values):
            print(max(v))
            xdata = np.arange(1000, (len(v) + 1) * 1000, 1000)
            yhat = savgol_filter(v, 51, 3)
            create_plot(xdata, yhat, i, test_names[i])
        plt.show()
    else:
        print('No values found.')

if __name__ == '__main__':
    main()
