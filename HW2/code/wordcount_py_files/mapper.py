#!/bin/python3
import sys

def write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")

for line in sys.stdin:
    words = line.strip().split()
    for w in words:
        write(w, "1")
