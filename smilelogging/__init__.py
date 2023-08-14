import configargparse
import functools
import sys

from smilelogging.logger import Logger
from smilelogging.slutils import update_args, red, green, yellow, blue

argparser = configargparse.ArgumentParser()
argparser.add_argument('--experiment_name',
                       type=str,
                       default='',
                       help='experiment name')
argparser.add_argument('--experiments_dir',
                       type=str,
                       default='Experiments',
                       help='name of the folder to store all experiments')
argparser.add_argument('--debug',
                       action="store_true",
                       help='if so, all the logs will be saved to `Debug_Dir`')
argparser.add_argument('--no_cache',
                       action='store_true',
                       help='not cache code')
argparser.add_argument('--cache_code',
                       type=str,
                       default='scripts/cache_code.sh',
                       help='the script to cache code')
argparser.add_argument('--no_scp',
                       action='store_true',
                       help='not scp experiment to hub')
argparser.add_argument('--resume_TimeID',
                       type=str,
                       default='',
                       help='the time ID used to uniquely identify an experiment')

# Customize smilelogging setups
argparser.add_argument('--sl.ON',
                       action='store_true')
argparser.add_argument('--sl.config',
                       type=str,
                       default='.smilelogging_cfg')

def warn_deprecated_args(old, new):
    if old in sys.argv:
        print(f'[Smilelogging Error] {old} is deprecated now, please use {new} instead and rerun')
        exit(0)

def add_update_args(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        ret = update_args(ret)
        return ret
    return wrapper

warn_deprecated_args('--project_name', '--experiment_name')
warn_deprecated_args('--project', '--experiment_name')
argparser.parse_args = add_update_args(argparser.parse_args)