from itertools import permutations, chain
from multiprocessing import Pool, cpu_count
import numpy as np
from colorama import Fore, Style, init

init()
DTYPE = np.dtype('i2')

class Draw:
    block = '█'

    @classmethod
    def yellow_block(cls):
        return Fore.YELLOW + cls.block

    @classmethod
    def blue_block(cls):
        return Fore.BLUE + cls.block

class Pattern:
    @staticmethod
    def _has_next(iterable):
        try:
            first = next(iterable)
        except StopIteration:
            return None
        return chain([first], iterable)

    @classmethod
    def _divisions(cls, num, space=float('inf'), lim=float('inf'), depth=0):
        result = []
        for chunk in range(min(num, lim), 0, -1):
            rchunk = num - chunk
            if rchunk == 0:
                result.append([chunk])
                continue
            if depth + 2 > space:
                continue
            rchunk_divisions = cls._divisions(
                rchunk, space, lim=chunk, depth=depth+1)
            for rest_chunk_division in rchunk_divisions:
                result.append([chunk, *rest_chunk_division])
        return result

    @classmethod
    def _division_with_filled_space(cls, space, num):
        for division in cls._divisions(num, space):
            while len(division) < space:
                division.append(0)
            yield division

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
        pt_set = np.zeros((1, length), dtype=DTYPE)
        wb_pt = cls._has_next(
            cls.combinations(S+1, W - (S - 1)))
        if wb_pt is None:
            pt = np.array([], dtype=DTYPE)
            for i, bsize in enumerate(key):
                pt = np.append(pt, np.ones(bsize))
                if i != len(key) - 1:
                    pt = np.append(pt, -1)
            pt_set = np.array([pt], dtype=DTYPE)
        else:
            for wb in wb_pt:
                pt = np.array([], dtype=DTYPE)
                for i, wsize in enumerate(wb):
                    pt = np.append(pt, np.full(wsize, -1, dtype=DTYPE))
                    if i < len(key):
                        pt = np.append(pt, np.ones(key[i], dtype=DTYPE))
                        if i != len(key) - 1:
                            pt = np.append(pt, -1)
                if pt_set[0, 0] == 0:
                    pt_set = np.array([pt], dtype=DTYPE)
                else:
                    pt_set = np.append(pt_set, np.array(
                        [pt], dtype=DTYPE), axis=0)
        return pt_set


class Nonogram:
    def __init__(self, row_keys, col_keys, processes=None):
        self.solved = False
        self.row_keys = row_keys
        self.col_keys = col_keys
        self.row = len(self.row_keys)
        self.col = len(self.col_keys)
        self.coordinate = np.zeros((self.row, self.col), dtype=DTYPE)
        pt_arg = [(row_keys[i], self.row) for i in range(self.row)] + \
            [(col_keys[i], self.col) for i in range(self.col)]

        proc_ct = self._processes_policy() if processes is None else processes
        with Pool(processes=min(cpu_count(), proc_ct)) as pool:
            all_pt = pool.map(Pattern.patterns_from_map, pt_arg)
            self.row_pt = all_pt[:self.row]
            self.col_pt = all_pt[self.row:]

    def _processes_policy(self):
        longest = max(self.row, self.col)
        if longest <= 10:
            return 1
        elif longest <= 15:
            return 3
        elif longest <= 25:
            return 5
        else:
            return cpu_count()

    def status(self):
        percentage = round((np.count_nonzero(self.coordinate) /
                            (self.row * self.col)) * 100, 1)
        print(f'[{self.row}X{self.col}] coordinate({percentage}%):')
        print(self.coordinate)

    def consensus(self, pt):
        thresh = pt.shape[0]
        def check_B(x): return 1 * (x == thresh)
        def check_W(x): return -1 * (x == -thresh)
        dist = np.sum(pt, axis=0)
        return check_B(dist) + check_W(dist)

    def sync_consensus_row(self):
        for i in range(self.row):
            self.coordinate[i] = np.bitwise_or(
                self.coordinate[i], self.consensus(self.row_pt[i]))

    def sync_consensus_col(self):
        for i in range(self.col):
            self.coordinate[:, i] = np.bitwise_or(
                self.coordinate[:, i], self.consensus(self.col_pt[i]))

    def sync_patterns_row(self):
        for i in range(self.row):
            nonzero_indices = np.nonzero(self.coordinate[i])[0]
            confirmation = np.take(self.coordinate[i], nonzero_indices)
            inconsistency = [j for j, pt in enumerate(self.row_pt[i]) 
                if not np.array_equal(confirmation, np.take(pt, nonzero_indices))]
            self.row_pt[i] = np.delete(self.row_pt[i], inconsistency, axis=0)

    def sync_patterns_col(self):
        for i in range(self.col):
            nonzero_indices = np.nonzero(self.coordinate[:, i])[0]
            confirmation = np.take(self.coordinate[:, i], nonzero_indices)
            inconsistency = [j for j, pt in enumerate(self.col_pt[i]) 
                if not np.array_equal(confirmation, np.take(pt, nonzero_indices))]
            self.col_pt[i] = np.delete(self.col_pt[i], inconsistency, axis=0)
    
    def solve(self):
        self.sync_consensus_row()
        self.sync_consensus_col()
        nonzero = np.count_nonzero(self.coordinate)
        while not self.verify():
            self.sync_patterns_col()
            self.sync_patterns_row()
            self.sync_consensus_row()
            self.sync_consensus_col()
            tmp = np.count_nonzero(self.coordinate)
            if nonzero == tmp:
                print("I can't solve this anymore...")
                self.status()
                break 
            else:
                nonzero = tmp
        else:
            self.solved = True
    
    def draw(self):
        if self.solved:
            base = np.where(self.coordinate == 1, Draw.blue_block(), Draw.yellow_block())
            for b in base:
                print(''.join(b))
        else:
            print('Nonogram is not solved yet...')

    def verify(self):
        return False if 0 in self.coordinate else True

def parse_from_text(keystxt):
    def parse(keytxt):
        return tuple(tuple(map(int, _.split(' ')))
                     for _ in keytxt.split(';'))
    keytxt = keystxt.split('\n')
    return parse(keytxt[0]), parse(keytxt[1])

def parse_from_file(filename):
    with open(filename) as f:
        return parse_from_text(f.read())
