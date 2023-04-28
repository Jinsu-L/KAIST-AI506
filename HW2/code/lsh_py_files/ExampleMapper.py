#!/bin/python3
import sys

def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")

def _map(key, value):
    _write(key, "\t".join(value)) # example

for line in sys.stdin:
    tokens = line.strip().split("\t")
    # key, value = tokens[0], tokens[1:]
    _map(tokens[0], tokens[1:])
