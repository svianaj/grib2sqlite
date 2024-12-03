from .__init__ import * 

"""Command line version."""
import argparse
from glob import glob
import json
import pandas 
parser = argparse.ArgumentParser(prog="extractsqlite")
current_wdir = os.getcwd()
# positional argument:
# we allow multiple GRIB files, but some might even be globs
parser.add_argument("gribfile", metavar="/path/to/file(s)",
        nargs='+' )
parser.add_argument('-p', metavar="param_file",
    help="parameter list file (json)",
    default = basedir + "/data/param_list_default.json")
parser.add_argument('-s', metavar="station_file",
    help="station list file (csv)",
    default = basedir + "/data/station_list_default.csv")
parser.add_argument('-m', metavar="model_name",
    help="model name used in SQLite (usually <*>_det)",
    default = "TEST_det")
parser.add_argument('-t', metavar="output_template",
    help="template used for SQLite files",
    default = "{MODEL}/{YYYY}/{MM}/FCTABLE_{PP}_{YYYY}{MM}.sqlite")
parser.add_argument('-o', metavar="output_path",
    help="path used for SQLite files",
    default = current_wdir + "/FCTABLE")
parser.add_argument('-d', metavar="debug_level",
    type = int,
    help="verbosity level (0...3) default: 1",
    default = 1)

args = parser.parse_args()
if args.d == 0:
    logger.setLevel('ERROR')
elif args.d == 1:
    logger.setLevel('WARNING')
elif args.d == 2:
    logger.setLevel('INFO')
elif args.d > 2:
    logger.setLevel('DEBUG')

station_file = args.s
param_file = args.p
model_name = args.m
infile = args.gribfile
sqlite_template = args.o + "/" + args.t  # "./@PP@_test.sqlite"

logger.debug(f"Binary path    : {basedir} ")

logger.info(f"GRIB file      : {infile}")
logger.info(f"Parameter list : {param_file}")
logger.info(f"Station list   : {station_file}")
logger.info(f"Model name     : {model_name}")
logger.info(f"Output files   : {sqlite_template}")

if len(infile) == 0:
    logger.error("No input GRIB file given.")
    exit()

logger.debug(f"Total file input: {infile}")
iter = 0
for file_glob in infile:
    logger.debug(f"file_glob = {file_glob}")
    file_list = glob(file_glob)
    # FIXME: this turns a non-existing file name into an empty list of files...
    #        so we may loose the error message
    # FIXME 2 : what should happen if the first file is missing, but a second does exist?
    #           Currently, it will fail. Probably what we want.
    logger.debug(f"file_list = {file_list}")
    if len(file_list) == 0:
        logger.error("No input files found!")
        raise FileNotFoundError("No input files found!")
    for file_name in file_list:
        logger.debug(f"file_name = {file_name}")
        iter = iter + 1
        logger.info(f"{iter:3d}:  Reading {file_name}")
        if not os.path.isfile(file_name):
            logger.warning(f"GRIB file not found: {file_name}")
            continue
        gt, gi, gd, gc = parse_grib_file(
            infile = file_name,
            param_list = param_file,
            station_list = station_file,
            sqlite_template = sqlite_template,
            model_name = model_name,
            weights = None)
        logger.info(f"Found {gt} grib messages, of which {gi} matching.")
        logger.info(f"Found {gd} direct parameters, {gc} combined.")


