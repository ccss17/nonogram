from itertools import permutations
import numpy as np
from config import *


def full(num, count):
    return [num for _ in range(count)]


def zeros(count):
    return full(0, count)


def ones(count):
    return full(1, count)


def minus(count):
    return full(-1, count)


class Pattern:
    @classmethod
    def divisions(cls, num, space=float('inf'), lim=float('inf'), depth=0):
        for chunk in range(min(num, lim), 0, -1):
            rchunk = num - chunk
            if rchunk == 0:
                yield [chunk]
                continue
            if depth + 2 > space:
                continue
            rchunk_divisions = cls.divisions(
                rchunk, space, lim=chunk, depth=depth+1)
            for rest_chunk_division in rchunk_divisions:
                yield [chunk, *rest_chunk_division]

    @classmethod
    def _division_with_filled_space(cls, space, num):
        for division in cls.divisions(num, space):
            division += zeros(space - len(division))
            yield division

    @classmethod
    def combinations(cls, space, num):
        for v in cls._division_with_filled_space(space, num):
            for pattern in set(permutations(v)):
                yield pattern

    @classmethod
    def patterns_from_map(cls, arg):
        return cls.patterns(arg[0], arg[1])

    @classmethod
    def patterns(cls, key, length):
        S = len(key)    # number of keys
        B = sum(key)    # number of black cells
        W = length - B  # number of white cells
        VARIABLE_FACTOR = W - (S - 1)

        if not VARIABLE_FACTOR:
            pattern = []
            for i, black_cell in enumerate(key):
                pattern += ones(black_cell)
                if i != S - 1:
                    pattern.append(-1)
            return np.array([pattern], dtype=DTYPE)

        white_block_patterns = cls.combinations(S + 1, VARIABLE_FACTOR)
        pattern_set = np.zeros((1, length), dtype=DTYPE)
        for white_block_pattern in white_block_patterns:
            pattern = []
            for i, white_block in enumerate(white_block_pattern):
                pattern += minus(white_block)
                if i < S:
                    pattern += ones(key[i])
                    if i != S - 1:
                        pattern.append(-1)
            if pattern_set[0, 0] == 0:
                pattern_set = np.array([pattern], dtype=DTYPE)
            else:
                pattern_set = np.append(pattern_set, np.array(
                    [pattern], dtype=DTYPE), axis=0)
        return pattern_set
