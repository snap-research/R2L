import os
import socket
import copy
import re

from colorama import init, Fore, Back, Style  # Need to pip install colorama first

init()  # Initialize colorama


def get_exp_name_id(exp_path):
    r"""arg examples: Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318
            or Experiments/kd-vgg13vgg8-cifar100-Temp40_SERVER5-20200727-220318/weights/ckpt.pth
            or PruneCnst_AlignCnst__PR0.875__ddprun_RANK0-SERVER196-20230614-194146
    Args:
        exp_path: experiment path (can be the experiment folder or a ckpt)
    Returns:
        ExpID: SERVER5-20200727-220318
        expid: 220318
        expname: kd-vgg13vgg8-cifar100-Temp40
        date: 20200727
    """
    seps = exp_path.split(os.sep)
    try:
        for s in seps:
            if 'SERVER' in s:
                expid = s.split('-')[-1]
                assert expid.isdigit()
                ExpID = 'SERVER' + s.split('SERVER')[1]
                expname = s.split('-SERVER')[0] if '_RANK' in s else s.split('_SERVER')[0]
                date = s.split('-')[-2]
    except:
        print(f'Failed to parse "{exp_path}", please check')
        exit(0)
    return ExpID, expid, expname, date


def clean_colored_text(logs):
    # Define the pattern to match the escape sequences
    pattern = r'\x1b\[[0-9;]*[m]'

    # Use regex to find and remove escape sequences from the logs
    cleaned_logs = re.sub(pattern, '', logs)

    return cleaned_logs


def standardize_metricline(line):
    r"""Make metric line in standard form.
    """
    line = clean_colored_text(line)
    for m in ['(', ')', '[', ']', '<', '>', '|', ',', ';', '!', '?', ]:  # Some non-numerical, no-meaning marks
        if m in line:
            line = line.replace(m, f' {m} ')
    if ':' in line:
        line = line.replace(':', ' ')
    line = ' '.join(line.split())
    return line


def get_value(line, key, type_func=float):
    r"""Get the value of a <key> in <line> in a log txt.
    """
    # Preprocessing to deal with some non-standard log format
    line = standardize_metricline(line)

    value = line.split(f' {key} ')[1].strip().split()[0]

    # Manually fix some problems.
    if '/' in value:  # E.g., Epoch 199/200
        value = value.split('/')[0]

    if value.endswith('%'):
        value = type_func(value[:-1]) / 100.
    else:
        value = type_func(value)
    return value


def replace_value(line, key, new_value):
    line = standardize_metricline(line)
    value = line.split(key)[1].strip().split()[0]
    line = line.replace(f' {key} {value} ', f' {key} {new_value} ')
    return line


def get_project_name():
    cwd = os.getcwd()
    # assert '/Projects/' in cwd
    return cwd.split(os.sep)[-1]


# acc line example: Acc1 71.1200 Acc5 90.3800 Epoch 840 (after update) lr 5.0000000000000016e-05 (Best_Acc1 71.3500 @ Epoch 817)
# acc line example: Acc1 0.9195 @ Step 46600 (Best = 0.9208 @ Step 38200) lr 0.0001
# acc line example: ==> test acc = 0.7156 @ step 80000 (best = 0.7240 @ step 21300)
def is_metric_line(line, mark=''):
    r"""This function determines if a line is an accuracy line. Of course the accuracy line should meet some 
    format features which @mst used. So if these format features are changed, this func may not work.
    """
    line = standardize_metricline(line)
    if mark:
        return mark in line
    else:
        line = line.lower()
        return "acc" in line and "best" in line and 'lr' in line \
               and 'resume' not in line and 'finetune' not in line


def parse_metric(line, metric, scale=1.):
    r"""Parse out the metric value of interest.
    """
    line = line.strip()
    # Get the last metric
    try:
        metric_l = get_value(line, metric)
    except:
        print(f'Parsing last metric failed; please check! The line is "{line}"')
        exit(1)

    # Get the best metric
    try:
        if f'Best {metric}' in line:  # previous impl.
            metric_b = get_value(line, f'Best {metric}')
        elif f'Best_{metric}' in line:
            metric_b = get_value(line, f'Best_{metric}')
        elif f'Best{metric}' in line:
            metric_b = get_value(line, f'Best{metric}')
        else:
            metric_b = -1  # Not found the best metric value (not written in log)
    except:
        print(f'Parsing best metric failed; please check! The line is "{line}"')
        exit(1)
    return metric_l * scale, metric_b * scale


def parse_time(line):
    r"""Parse the time (e.g., epochs or steps) in a metric line.
    """
    line = standardize_metricline(line)
    if ' Epoch ' in line:
        time = get_value(line, 'Epoch', type_func=int)
    elif ' Step ' in line:
        time = get_value(line, 'Step', type_func=int)
    elif ' step ' in line:
        time = get_value(line, 'step', type_func=int)
    else:
        print(f'Fn "parse_time" failed. Please check')
        raise NotImplementedError
    return time


def parse_finish_time(log_f):
    lines = open(log_f, 'r').readlines()
    for k in range(1, min(1000, len(lines))):
        line = lines[-k].lower()
        if '(speed:' in line and 'per timing, total_time:' in line:
            finish_time = lines[-k].split('(speed:')[0].split()[
                -1].strip()  # example: predicted finish time: 2020/10/25-08:21 (speed: 314.98s per timing)
            return finish_time


def get_ip():
    # Get IP address. Refer to: https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


class EmptyClass():
    pass


def update_args(args):
    """Update arguments of configargparse
    """
    arg_dict = copy.deepcopy(args.__dict__)
    for k, v in arg_dict.items():
        if '.' in k:  # TODO-@mst: hardcode pattern, may be risky
            module, arg = k.split('.')  # e.g., "deepmixup.depth"
            if arg_dict[f'{module}.ON']:  # this module is being used
                if not hasattr(args, module):
                    args.__setattr__(module, EmptyClass())  # set to a blank class
                args.__dict__[module].__dict__[arg] = v  # args.'deepmixup.depth' = 10 --> args.deepmixup.depth = 10
            args.__delattr__(k)
    return args


def red(*msg, sep=','):
    """Wrap log string with red color"""
    msg = sep.join([str(x) for x in msg])
    return Fore.RED + msg + Style.RESET_ALL


def green(*msg, sep=','):
    """Wrap log string with green color"""
    msg = sep.join([str(x) for x in msg])
    return Fore.GREEN + msg + Style.RESET_ALL


def yellow(*msg, sep=','):
    """Wrap log string with yellow color"""
    msg = sep.join([str(x) for x in msg])
    return Fore.YELLOW + msg + Style.RESET_ALL


def blue(*msg, sep=','):
    """Wrap log string with blue color"""
    msg = sep.join([str(x) for x in msg])
    return Fore.BLUE + msg + Style.RESET_ALL
