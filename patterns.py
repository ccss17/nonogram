import itertools
import numpy as np
from config import *
from time import time


def full(num, count):
    return [num] * count


def zeros(count):
    return full(0, count)


def ones(count):
    return full(1, count)


def minus(count):
    return full(-1, count)


class Pattern:
    @classmethod
    def divisions(cls, num, space=float('inf'), lim=float('inf'), depth=0, fill_space=False):
        for main_chunk in range(min(num, lim), 0, -1):
            rest_chunk = num - main_chunk
            if rest_chunk == 0:
                if fill_space and depth == 0:
                    yield [main_chunk] + zeros(space - 1)
                else:
                    yield [main_chunk]
                continue
            if depth + 2 > space:
                continue
            rchunk_divisions = cls.divisions(
                rest_chunk, space, lim=main_chunk, depth=depth+1)
            for rest_chunk_division in rchunk_divisions:
                if fill_space and depth == 0:
                    yield [main_chunk, *rest_chunk_division] + zeros(space - (1 + len(rest_chunk_division)))
                else:
                    yield [main_chunk, *rest_chunk_division]

    @classmethod
    def permutations_without_duplication(cls, iterable, l=0, r=None):
        r = len(iterable)-1 if r is None else r
        black_list = []
        for i in range(l, r+1):
            if (iterable[l] == iterable[i] and l != i) or\
               (iterable[i] in black_list):
                continue
            black_list.append(iterable[i])
            iterable[l], iterable[i] = iterable[i], iterable[l]
            if l+1 == r:
                yield tuple(iterable)
            else:
                for case in cls.permutations_without_duplication(iterable, l+1, r):
                    yield case
            iterable[l], iterable[i] = iterable[i], iterable[l]  # backtrack

    @classmethod
    def white_cell_patterns(cls, space, num):
        for v in cls.divisions(num, space, fill_space=True):
            for pattern in cls.permutations_without_duplication(v):
                yield pattern

    @classmethod
    def patterns_from_map(cls, arg):
        # arg[0] -> 'row' or 'col'
        # arg[1] -> index
        # arg[2] -> keys
        # arg[3] -> length
        start_time = time()
        result = cls.patterns(arg[2], arg[3])
        if sum(arg[2]) != 0:
            timedata = (time()-start_time, sum(arg[2]), len(arg[2]), len(arg[2])/sum(arg[2]))
        else:
            timedata = (time()-start_time, sum(arg[2]), len(arg[2]), 0)
        return arg[0], arg[1], result, timedata

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

        white_cell_patterns = cls.white_cell_patterns(S + 1, VARIABLE_FACTOR)
        pattern_set = None
        for white_cell_pattern in white_cell_patterns:
            pattern = []
            for i, white_cell in enumerate(white_cell_pattern):
                pattern += minus(white_cell)
                if i < S:
                    pattern += ones(key[i])
                    if i != S - 1:
                        pattern.append(-1)
            if pattern_set is None:
                pattern_set = np.array([pattern], dtype=DTYPE)
            else:
                pattern_set = np.append(pattern_set, np.array(
                    [pattern], dtype=DTYPE), axis=0)

        return pattern_set
