#!/bin/python3
import sys

s = 10

def hashFunction(func_id, first_word, second_word, third_word):
    # hashFunction(2, 'i', 'have', 'a') -> 2nd hash function value of 3-shingle ('i', 'have', 'a')
    # 1 <= func_id <= 10
    def wordHash(s):
        ret = 0
        for c in s[::-1]:
            ret = ((ret * 31) + ((ord(c) - ord('a')) + 1)) % 1234567891
        return ret

    return ((41 ** func_id) * wordHash(first_word) * (2 ** 64) + (506 ** func_id) * wordHash(second_word) * (
            2 ** 32) + wordHash(third_word)) % 1234567891


def _write(key, value):
    sys.stdout.write(f"{key}\t{value}\n")


# paper_id, 3-shingle
def _map(key, value):
    # _write(key, "\t".join(value)) # example
    first, second, third = value[0].split(",")
    for i in range(1, s + 1):
        hashed_value = hashFunction(i, first, second, third)

        # key : paper_id, vale : hashidx_hashed_value
        _write(key, ",".join([str(i), str(hashed_value)]))
    # signature = tuple([hashFunction(i, first, second, third) for i in range(1, s + 10)])


for line in sys.stdin:
    tokens = line.strip().split("\t")
    # key, value = tokens[0], tokens[1:]
    _map(tokens[0], tokens[1:])
