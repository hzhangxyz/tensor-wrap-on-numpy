#!/usr/bin/env python
from tnsp_np import SquareLattice
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--continue', dest='continued', default=False, action="store_true", help="continue run, conflict with -f")
parser.add_argument('-N', '--size-n', dest='n', required=True, type=int, help="system size n")
parser.add_argument('-M', '--size-m', dest='m', required=True, type=int, help="system size m")
parser.add_argument('-d', '--dim', dest='D', required=True, type=int, help="bond dimension")
parser.add_argument('-l', '--step-size', dest='step_size', required=True, type=float, help="step size in SU or GM")
parser.add_argument('-e', '--step-print', dest='step_print', required=True, type=int, help="how many step print once in SU")
parser.add_argument('-f', '--load-from', dest='load_from', help="load from file")
parser.add_argument('-p', '--save-prefix', dest='save_prefix', default='run', help="prefix for saving data")
parser.add_argument('-a', '--energy', dest='energy', default=False, action="store_true", help="calculate energy")
args = parser.parse_args()

if args.continued and args.load_from != None:
    exit("conflict between --continue and --load-from")

if args.continued and not args.load_from:
    args.load_from = f"{args.save_prefix}/last/last.npz"

lattice = SquareLattice([args.n, args.m], args.D, args.step_size, args.load_from, args.save_prefix, args.step_print)
lattice.itebd(args.energy)
