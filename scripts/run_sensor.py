"""
TODO
"""

import os
import time

from sbrl.experiments import logger
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.python_utils import exit_on_ctrl_c
from sbrl.utils.script_utils import load_standard_ml_config

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str)
grouped_parser.add_argument('--dt', type=float, default=0.)
grouped_parser.add_argument('--measure_hz', action='store_true')
local_args, unknown = grouped_parser.parse_local_args()
# this defines the required command line groups, and the defaults
# if in this list, we look for it

ordered_modules = ['sensor']

config_fname = os.path.abspath(local_args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
exp_name, args, params = load_standard_ml_config(config_fname, unknown, grouped_parser, ordered_modules,
                                                 debug=False)


sensor = params.sensor.cls(params.sensor.params)
sensor.open()

count = 0
last_hz_time = time.time()

while True:
    obs = sensor.read_state()
    time.sleep(local_args.dt)

    count += 1
    if time.time() - last_hz_time > 2.0:
        logger.info(f"Avg. Hz -> {count / (time.time() - last_hz_time)}")
        count = 0
        last_hz_time = time.time()

# sensor.close()
