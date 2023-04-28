#!/bin/python3
import sys

def write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")

curr_word, curr_count = None, 0

for line in sys.stdin:
    key_from_map, value_from_map = line.strip().split()
    if curr_word != key_from_map:
        if curr_count:
            write(curr_word, curr_count)
        curr_word = key_from_map
        curr_count = 0
    curr_count += int(value_from_map)

if curr_count:
    write(curr_word, curr_count)
