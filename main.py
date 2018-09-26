#!/usr/bin/env python
from square_lattice import SquareLattice as SL
import numpy_wrap as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-S', '--simple-update', dest='update', default=False, action="store_true", help="run simple update instead of gradient descent")
parser.add_argument('-c', '--continue', dest='continued', default=False, action="store_true", help="continue run, conflict with -f")
parser.add_argument('-N', '--size-n', dest='n', required=True, type=int, help="system size n")
parser.add_argument('-M', '--size-m', dest='m', required=True, type=int, help="system size m")
parser.add_argument('-d', '--dim', dest='D', required=True, type=int, help="bond dimension")
parser.add_argument('-D', '--dim-cut', dest='D_c', required=True, type=int, help="dimension cut in MPO")
parser.add_argument('-s', '--scan-time', dest='scan_time', required=True, type=int, help="scan time in MPO")
parser.add_argument('-l', '--step-size', dest='step_size', required=True, type=float, help="step size in SU or GM")
parser.add_argument('-m', '--markov', dest='markov', required=True, type=int, help="markov chain length")
parser.add_argument('-f', '--load-from', dest='load_from', help="load from file")
parser.add_argument('-p', '--save-prefix', dest='save_prefix', default='run', help="prefix for saving data")
parser.add_argument('-e', '--step-print', dest='step_print', type=int, help="how many step print once in SU")
args = parser.parse_args()

if args.continued and args.load_from != None:
    exit("conflict between --continue and --load-from")

if args.update and args.step_print == None:
    exit("--step-print needed when simple update")

if args.continued and not args.load_from:
    args.load_from = f"{args.save_prefix}/last/last.npz"

if args.update:
    lattice = SL(args.n, args.m, args.D, args.D_c, args.scan_time, args.step_size, args.markov, args.load_from, args.save_prefix, args.step_print)
    lattice.itebd()
else:
    lattice = SL(args.n, args.m, args.D, args.D_c, args.scan_time, args.step_size, args.markov, args.load_from, args.save_prefix)
    lattice.grad_descent()
