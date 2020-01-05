from multiprocessing import Pool, cpu_count
import numpy as np
from colorama import Fore, Style, init
from patterns import *
from config import *

init()


class Draw:
    block = '█'
    yellow_block = Fore.YELLOW + block
    blue_block = Fore.BLUE + block
    white_block = Fore.WHITE + block


class Nonogram:
    def __init__(self, row_keys, col_keys, processes=None):
        self.solved = False
        self.row_keys = row_keys
        self.col_keys = col_keys
        self.row = len(self.row_keys)
        self.col = len(self.col_keys)
        self.coordinate = np.zeros((self.row, self.col), dtype=DTYPE)
        self.row_patterns = []
        self.col_patterns = []
        self.processes = processes

    def init_patterns(self):
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

        proc_count = processes_policy() if self.processes is None else self.processes
        with Pool(processes=min(cpu_count(), proc_count)) as pool:
            arg = [(self.row_keys[i], self.row) for i in range(self.row)] + \
                [(self.col_keys[i], self.col) for i in range(self.col)]
            all_patterns = pool.map(Pattern.patterns_from_map, arg)
            self.row_patterns = all_patterns[:self.row]
            self.col_patterns = all_patterns[self.row:]

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
                self.coordinate[i], self.consensus(self.row_patterns[i]))
        for i in range(self.col):
            self.coordinate[:, i] = np.bitwise_or(
                self.coordinate[:, i], self.consensus(self.col_patterns[i]))

    def sync_patterns(self):
        for i in range(self.row):
            if not np.count_nonzero(self.coordinate[i]) == self.row:
                self.row_patterns[i] = self.consistent_patterns(
                    self.row_patterns[i], self.coordinate[i])
        for i in range(self.col):
            if not np.count_nonzero(self.coordinate[:, i]) == self.col:
                self.col_patterns[i] = self.consistent_patterns(
                    self.col_patterns[i], self.coordinate[:, i])

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
