#!/usr/bin/env python3 

import argparse
import json 
import os, sys 

# --- version import fallback ---
try:
    from importlib.metadata import version as _pkg_version
    JPLOT_VERSION = _pkg_version("jarvisplot")
except Exception:
    JPLOT_VERSION = "0.0.0"

class CLI():
    def __init__(self):
        self.args = argparse.ArgumentParser(description="JarvisPLOT Help Center", formatter_class=argparse.RawTextHelpFormatter)
        self.pwd  = os.path.abspath(os.path.dirname(__file__))

        
        with open("{}/cards/args.json".format(self.pwd), 'r') as f1: 
            args = json.load(f1)
            # --- positionals, with nargs handling and 'file' optional by default ---
            for pos_arg in args.get("positionals", []):
                heltp = pos_arg['help'].replace("$n", "\n")
                pkwargs = {"help": heltp}
                # Allow JSON to set nargs; default: make 'file' optional so -v works without it
                if 'nargs' in pos_arg:
                    pkwargs['nargs'] = pos_arg['nargs']
                elif pos_arg.get('name') == 'file':
                    pkwargs['nargs'] = '?'
                self.args.add_argument(pos_arg['name'], **pkwargs)
            # --- options, handle version action with JarvisPLOT version ---
            for opt in args.get("options", []):
                action = opt.get('action', 'store')
                kwargs = {
                    'help': opt['help'],
                    'dest': opt.get('dest')
                }
                if action == 'version':
                    kwargs['action'] = 'version'
                    kwargs['version'] = f"JarvisPLOT {JPLOT_VERSION}"
                else:
                    kwargs['action'] = action
                    if 'metavar' in opt:
                        kwargs['metavar'] = opt['metavar']
                if 'default' in opt:
                    kwargs['default'] = opt['default']
                if "type" in opt:
                    if opt['type'] == 'int':
                        kwargs['type'] = int
                    elif opt['type'] == 'float':
                        kwargs['type'] = float
                    else:
                        kwargs['type'] = str
                if 'short' in opt and 'long' in opt:
                    self.args.add_argument(
                        opt['short'],
                        opt['long'],
                        **kwargs
                    )
                elif 'long' in opt:
                    self.args.add_argument(
                        opt['long'], **kwargs
                    )

