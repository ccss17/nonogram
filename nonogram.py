from itertools import permutations, chain
from multiprocessing import Pool, cpu_count
import numpy as np
from colorama import Fore, Style, init

init()
INT8 = np.int8
UBYTE = np.uint8


class Draw:
    block = '█'
    yellow_block = Fore.YELLOW + block
    blue_block = Fore.BLUE + block
    white_block = Fore.WHITE + block


class Pattern:
    @staticmethod
    def _has_next(iterable):
        try:
            first = next(iterable)
        except StopIteration:
            return None
        return chain([first], iterable)

    @classmethod
    def divisions(cls, num, space=float('inf'), lim=float('inf'), depth=0):
        for chunk in range(min(num, lim), 0, -1):
            rchunk = num - chunk
            if rchunk == 0:
                yield np.asarray([chunk], dtype=UBYTE)
                continue
            if depth + 2 > space:
                continue
            rchunk_divisions = cls.divisions(
                rchunk, space, lim=chunk, depth=depth+1)
            for rest_chunk_division in rchunk_divisions:
                yield np.asarray([chunk, *rest_chunk_division], dtype=UBYTE)

    @classmethod
    def _division_with_filled_space(cls, space, num):
        for division in cls.divisions(num, space):
            yield np.pad(division, (0, space - len(division)))

    @classmethod
    def combinations(cls, space, num):
        for v in cls._division_with_filled_space(space, num):
            for pt in set(permutations(v)):
                yield pt

    @classmethod
    def patterns_from_map(cls, arg):
        return cls.patterns(arg[0], arg[1])

    @classmethod
    def patterns(cls, key, length):
        S = len(key)
        B = sum(key)
        W = length - B
        white_block_patterns = cls._has_next(
            cls.combinations(S+1, W - (S - 1)))
        if white_block_patterns is None:
            pattern = np.array([], dtype=INT8)
            for i, bsize in enumerate(key):
                pattern = np.append(pattern, np.ones(bsize))
                if i != len(key) - 1:
                    pattern = np.append(pattern, -1)
            return np.array([pattern], dtype=INT8)
        else:
            pattern_set = np.zeros((1, length), dtype=INT8)
            for wb in white_block_patterns:
                pattern = np.array([], dtype=INT8)
                for i, wsize in enumerate(wb):
                    pattern = np.append(
                        pattern, np.full(wsize, -1, dtype=INT8))
                    if i < len(key):
                        pattern = np.append(
                            pattern, np.ones(key[i], dtype=INT8))
                        if i != len(key) - 1:
                            pattern = np.append(pattern, -1)
                if pattern_set[0, 0] == 0:
                    pattern_set = np.array([pattern], dtype=INT8)
                else:
                    pattern_set = np.append(pattern_set, np.array(
                        [pattern], dtype=INT8), axis=0)
            return pattern_set


class Nonogram:
    def __init__(self, row_keys, col_keys, processes=None):
        self.solved = False
        self.row_keys = row_keys
        self.col_keys = col_keys
        self.row = len(self.row_keys)
        self.col = len(self.col_keys)
        self.coordinate = np.zeros((self.row, self.col), dtype=INT8)
        def processes_policy():
            longest = max(self.row, self.col)
            if longest <= 10:
                return 1
            elif longest <= 15:
                return 3
            elif longest <= 25:
                return 5
            else:
                return cpu_count()

        proc_ct = processes_policy() if processes is None else processes
        with Pool(processes=min(cpu_count(), proc_ct)) as pool:
            pt_arg = [(row_keys[i], self.row) for i in range(self.row)] + \
                [(col_keys[i], self.col) for i in range(self.col)]
            all_pt = pool.map(Pattern.patterns_from_map, pt_arg)
            self.row_pt = all_pt[:self.row]
            self.col_pt = all_pt[self.row:]

    def consensus(self, pattern):
        thresh = pattern.shape[0]
        def check_B(x): return 1 * (x == thresh)
        def check_W(x): return -1 * (x == -thresh)
        dist = np.sum(pattern, axis=0)
        return check_B(dist) + check_W(dist)

    def consistent_patterns(self, pattern_set, line):
        nonzero_indices = np.nonzero(line)[0]
        confirmation = line[nonzero_indices]
        consistency = []
        for i, pattern in enumerate(pattern_set):
            if np.array_equal(confirmation, pattern[nonzero_indices]):
                consistency.append(i)
        return pattern_set[consistency]

    def status(self):
        percentage = round((np.count_nonzero(self.coordinate) /
                            (self.row * self.col)) * 100, 1)
        print(f'[{self.row}X{self.col}] coordinate({percentage}%):')
        print(self.coordinate)

    def draw(self):
        if self.solved:
            base = np.where(self.coordinate == 1,
                            Draw.blue_block, Draw.white_block)
            for b in base:
                print(''.join(b))
        else:
            print('Nonogram is not solved yet...')

class NonogramHacker(Nonogram):
    def verify(self):
        return False if 0 in self.coordinate else True

    def sync_consensus(self):
        for i in range(self.row):
            self.coordinate[i] = np.bitwise_or(
                self.coordinate[i], self.consensus(self.row_pt[i]))
        for i in range(self.col):
            self.coordinate[:, i] = np.bitwise_or(
                self.coordinate[:, i], self.consensus(self.col_pt[i]))

    def sync_patterns(self):
        for i in range(self.row):
            if not np.count_nonzero(self.coordinate[i]) == self.row:
                self.row_pt[i] = self.consistent_patterns(
                    self.row_pt[i], self.coordinate[i])
        for i in range(self.col):
            if not np.count_nonzero(self.coordinate[:, i]) == self.col:
                self.col_pt[i] = self.consistent_patterns(
                    self.col_pt[i], self.coordinate[:, i])

    def solve(self):
        self.sync_consensus()
        nonzero = np.count_nonzero(self.coordinate)
        while not self.verify():
            self.sync_patterns()
            self.sync_consensus()
            tmp = np.count_nonzero(self.coordinate)
            if nonzero == tmp:
                print("I can't solve this anymore...")
                self.status()
                break
            else:
                nonzero = tmp
        else:
            self.solved = True


def parse_from_text(keystxt):
    def parse(keytxt):
        return tuple(tuple(map(int, _.split(' ')))
                     for _ in keytxt.split(';'))
    keytxt = keystxt.split('\n')
    return parse(keytxt[0]), parse(keytxt[1])


def parse_from_file(filename):
    with open(filename) as f:
        return parse_from_text(f.read())
