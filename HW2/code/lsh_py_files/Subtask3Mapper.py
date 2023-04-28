#!/bin/python3
import sys

b = 5
r = 2

def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")


# key = each bands vector, value = paper id
def _map(key, value):
    for vv in value:
        signatures = vv.split(",")
        for i in range(b):
            pair_key = [signatures[i * r + j] for j in range(r)]
            _write(",".join(pair_key), key)


for line in sys.stdin:
    tokens = line.strip().split("\t")
    # key, value = tokens[0], tokens[1:]
    _map(tokens[0], tokens[1:])
