import itertools 
import numpy as np
from config import *
from time import time


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
    def division_with_filled_space(cls, space, num):
        for division in cls.divisions(num, space):
            division += zeros(space - len(division))
            yield division
    
    @classmethod
    def permutations(cls, iterable, l=0, r=None): 
        r = len(iterable)-1 if r is None else r
        for i in range(l,r+1): 
            if iterable[l] == iterable[i] and l != i:
                continue
            iterable[l], iterable[i] = iterable[i], iterable[l] 
            if l+1 == r:
                yield tuple(iterable)
            else:
                for case in cls.permutations(iterable, l+1, r):
                    yield case
            iterable[l], iterable[i] = iterable[i], iterable[l] # backtrack 

    @classmethod
    def combinations(cls, space, num):
        for v in cls.division_with_filled_space(space, num):
            # for pattern in set(itertools.permutations(v)):
            for pattern in cls.permutations(v):
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

        # start_time = time()

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

        # elap_time = round(time()-start_time, 5)
        # if elap_time > 0.9:
        #     print('\t', key, length)
        #     print(f'\tPhase 1 Time Taken:{elap_time} secs')
        # elif elap_time < 0.01:
        #     pass
        # else:
        #     print(f'{elap_time} secs', end=' ')
        #     print(key, length)

        return pattern_set
