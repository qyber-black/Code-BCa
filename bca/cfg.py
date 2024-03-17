# bca/cfg.py - Configuration class
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import matplotlib.pyplot as plt
import json

class Cfg:
  """Class holding BCa's configuration options.

  This class holds and generated BCa's configuration options, via json files and environment variables.
  Any new configuration options will be added to the json files, while unused ones are removed, when
  it's called. To use it in a file, import it as

  ```from bca.cfg import Cfg```

  This will setup the configuration and it can be accessed via `Cfg`. Note that all values are in the class,
  (static), not in any object.

  It uses the json files `cfg.json` in the bca root directory and `~/.config/bca.json`; `cfg.json` overwrites
  `~/.config/bca.json` and default configuration from the sources are only used if they are not in the json file
  (and then they are added to it).

  For the configuration options in the json values and their defaults see `Cfg.val` in the sources.

  In addition the environment variable `BCA_DEV` can be used to set specific behaviour of the code, often only
  useful for development. See `Cfg.dev` in the sources for what they are.
  """

  def __init__(self):
    """Constructor - do not call.
    
    The constructor of the `Cfg` class should never be called as it hold all values in the class. Use the
    static methods of the class only.
    """
    raise Exception("Tried to create a Cfg objects; values should only be held in the class")

  # Log levels
  Error = 1
  Warn = 2
  Info = 3

  @staticmethod
  def log(level):
    """Return `True` if the configured verbose level is at least `level`

    Args:
      * `level`: Minimum log-level (`Cfg.Error`, `Cfg.Warn`, or `Cfg.Info`)

    Return:
      * `True` if log-level indicates message should be printed; `False` otherwise.

    Example:
      ```python
      if Cfg.log(Cfg.Warn):
        print("Warning: something went wrong")
      ```
    """
    return level <= Cfg.val["log"]

  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  #
  # Change these values in ROOT_PATH/cfg.json (generated after first run; overwrites
  # defaults here) or ~/.config/bca.json (not generated; overwrites cfg.json and
  # defaults here). Do not edit defaults here; will not work once cfg file is generated.
  val = {
    'path_root': None,
    'parallel': 2,              # Number of parallel threads for some tasks (via joblib)
    'multiprocessing': True,    # Use multiprocessing (way not work on windows, so disable here)
    'low_mem_percentage': 90,   # Percentage of memory indicating low CPU memory
    'brain_cmap': 'gray',       # Default color map for brain images
    'gt_cmap': 'gray',          # Ground truth colormap for output masks
    'pr_cmap': 'viridis',       # Prediction colormap for output masks
    'col_label0': [0,0,0,1],    # Label colors for masks
    'col_label1': [1,0,0,1],
    'col_label2': [0,1,0,1],
    'col_label3': [1,0,1,1],
    'col_label4': [0,0,1,1],
    'figsize': (5.0,5.0),       # Sub-figure size
    'default_screen_dpi': 96,   # Image resolution for display (default, used if estimation fails)
    'screen_dpi': None,
    'image_dpi': [300],         # Image resolution for saving 
    'log': 1,                   # Level of log messages printed (Cfg.{Error,Warn,Info} = 1,2,3)
    'tf_log': 'ERROR',          # Tensorflow logging level
    'executors': {              # List of executors for scheduler (configure in cfg.json)
      #'scw': {                   # Example slurm cluster
      #  'type': 'slurm',
      #  'max_tasks': 10,
      #  'host': 'hawklogin.cf.ac.uk',
      #  'user': 'USERNAME',
      #  'account': 'ACCOUNT',
      #  'remote_folder': 'code-bca',
      #  'partitions': "gpu_v100,gpu",
      #  'nodes': 1,
      #  'ntasks': 1,
      #  'ntasks_per_node': 1,
      #  'cpus_per_task': 4,
      #  'mem': '96G',
      #  'gres': 'gpu:2',
      #  'time': '2-00:00:00',
      #  'modules': [ "system/auto", "hpcw", "python/3.10.4", "load CUDA/11.5"]
      #},
      #'localhost': {             # Example host node (don't use localhost; see local type)
      #  'type': 'host',
      #  'host': 'localhost',
      #  'user': 'USERNAME',
      #  'remote_folder': 'exec-bca',
      #},
      'local': {                # Local node (made this the default, but probably needs editing)
        'type': 'local'
      }
    },
    "disabled_executors": []  # Short-cut to disable executors if they should not be used
  }
  # Development flags for extra functionalities and test (not relevant for use).
  # These are set via the environment variable BCA_DEV (colon separated list),
  # but all in use should be in the comments here for reference:
  #  FIXED_BATCH_SIZE: used in dataset.py by SeqGen to fix the batch_size (cuts off "modulo" samples)
  dev_flags = set()
  file = os.path.expanduser(os.path.join('~','.config','bca.json'))

  @staticmethod
  def init(bin_path):
    """Initialize the configuration values as static values of the class.

    This is used mostly internally to initialise the class (will be called upon import).

    Args:
      * `bin_path`: path to the foler containing the `cfg.py` file
    """
    # Root path of bca
    Cfg.val["path_root"] = os.path.dirname(bin_path)
    # Load cfg file - data folders and other Cfg values can be overwritten by config file
    # We first load ROOT/cfg.json, if it exists, then the user config file
    root_cfg_file = os.path.join(Cfg.val["path_root"],'cfg.json')
    root_cfg_vals = {}
    for fc in [root_cfg_file, Cfg.file]:
      if os.path.isfile(fc):
        with open(fc, "r") as fp:
          js = json.load(fp)
          if fc == root_cfg_file:
            root_cfg_vals = js
          for k in js.keys():
            if k in Cfg.val:
              Cfg.val[k] = js[k]
            else:
              if fc != root_cfg_file: # We fix this here later
                raise Exception(f"Unknown config file entry {k} in {fc}")
    # Setup plot defaults
    if Cfg.val["screen_dpi"] == None:
      Cfg.val["screen_dpi"] = Cfg._screen_dpi()
    plt.rcParams["figure.figsize"] = Cfg.val['figsize']
    # Store configs in ROOT/cfg.json if it does not exist
    changed = False
    del_keys = []
    for k in root_cfg_vals.keys(): # Do not store paths and remove old values
      if k[0:5] == 'path_' or k not in Cfg.val:
        del_keys.append(k)
        changed = True
    for k in del_keys:
      del root_cfg_vals[k]
    for k in Cfg.val: # Add any new values (except paths)
      if k[0:5] != 'path_' and k not in root_cfg_vals:
        root_cfg_vals[k] = Cfg.val[k]
        changed = True
    if changed:
      with open(root_cfg_file, "w") as fp:
        print(json.dumps(root_cfg_vals, indent=2, sort_keys=True), file=fp)
    # Dev flags
    if 'BCA_DEV' in os.environ:
      for f in os.environ['BCA_DEV'].split(":"):
        Cfg.dev_flags.add(f)

  @staticmethod
  def dev(flag):
    """Check if the developmnet flag `flag` has been set.

    See the sources for available flags. If you add any, make sure to add them to the sources!

    Args:
      * `flag`: string containing the development flag.
    """
    # Development flags for custom code behaviour; set via BCA_DEV environment variable
    return flag in Cfg.dev_flags

  @staticmethod
  def _screen_dpi():
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    try:
      m = get_monitors()[0]
    except:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    try:
      dpi = hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4
      return dpi # set in cfg.json if this is not working
    except:
      return Cfg.val['default_screen_dpi']

# TF log-level default
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# Find base folder
Cfg.init(os.path.dirname(os.path.realpath(__file__)))