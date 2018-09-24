#!/usr/bin/env python
import square_lattice as sl
import numpy_wrap as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--size-n', dest='n', required=True, type=int)
parser.add_argument('-M', '--size-m', dest='m', required=True, type=int)
parser.add_argument('-d', '--dim', dest='D', required=True, type=int)
parser.add_argument('-D', '--dim-cut', dest='D_c', required=True, type=int)
parser.add_argument('-t', '--scan-time', dest='scan_time', required=True, type=int)
parser.add_argument('-s', '--step-size', dest='step_size', required=True, type=float)
parser.add_argument('-m', '--markov-chain-length', dest='markov_chain_length', required=True, type=int)
parser.add_argument('-f', '--load-from', dest='load_from')
parser.add_argument('-p', '--save-prefix', dest='save_prefix')
parser.add_argument('-c', '--continue', dest='continued', default=False, action="store_true")
args = parser.parse_args()

if args.continued and not args.load_from:
    args.load_from = "./run/last/last.npz"

np.random.seed(0)

lattice = sl.SquareLattice(args.n, args.m, args.D, args.D_c, args.scan_time, args.step_size, args.markov_chain_length, args.load_from, args.save_prefix)

lattice.grad_descent()
