#!/bin/python3
import sys

def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")

def _reduce(key, values):
    pass

curr_key = None
curr_values = []
for line in sys.stdin:
    tokens = line.strip().split("\t")
    key, value = tokens[0], tokens[1:]
    if curr_key is not None and curr_key != key:
        _reduce(curr_key, curr_values)
        curr_values = []
    curr_key = key
    curr_values.append(value)

if curr_key is not None:
    _reduce(curr_key, curr_values)
