#!/bin/python3
import sys
from collections import defaultdict


def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")

# key : paper_id, vale : hashidx_hashed_value
def _reduce(key, values):
    collector = defaultdict(list)

    for value in values:
        for vv in value:
            tkn = vv.split(",")
            hash_idx, hashed_value = int(tkn[0]), int(tkn[1])
            collector[hash_idx].append(hashed_value)

    _write(key, ",".join([str(min(collector[i])) for i in range(1, 11)]))

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
