#!/bin/python3
import sys
import itertools

def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")


# key = ",".join(each bands vector), value = paper id
def _reduce(key, values):
    if len(values) > 1:
        docs = sorted([doc_id[0] for doc_id in values])
        for paper1, paper2 in itertools.combinations(docs, 2):
            _write(paper1, paper2)

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
