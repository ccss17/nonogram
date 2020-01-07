from multiprocessing import Pool, cpu_count
import numpy as np
from colorama import Fore, Style, init
from patterns import *
from config import *
import sys

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
        self.row_patterns = [None]*self.row
        self.col_patterns = [None]*self.col

        def _processes_policy(processes):
            proc = float('inf')
            limit = cpu_count() - 2
            if processes is None:
                longest = max(self.row, self.col)
                if longest <= 5:
                    processes = 1
                elif longest <= 10:
                    processes = 2
                elif longest <= 15:
                    processes = 3
                elif longest <= 25:
                    processes = 5
                else:
                    processes = limit
            return max(1, min(limit, processes))
        self.processes = _processes_policy(processes)
        self.init_patterns()

    def init_patterns(self):
        args = [('r', i, self.row_keys[i], self.row)
                for i in range(self.row)]
        args += [('c', i, self.col_keys[i], self.col)
                    for i in range(self.col)]
        tasks_count = len(args)
        # opp = {'r' : [i for i in range(self.row)]}
        # opp.update({'c' : [i for i in range(self.col)]})

        def optimal_chunksize(tasks_count, processes):
            d = 4
            chunksize = tasks_count // (self.processes * d)
            while chunksize == 0:
                chunksize = tasks_count // (self.processes * d)
                d -= 1
            return chunksize

        chunksize = optimal_chunksize(tasks_count, self.processes)
        with Pool(processes=self.processes) as pool:
            for i, result in enumerate(pool.imap_unordered(Pattern.patterns_from_map,
                                                           args,
                                                           chunksize=chunksize)):
                sys.stderr.write(
                    '\rCalculating patterns {0:.1%}'.format((i+1) / tasks_count))
                # result[0] -> 'r' or 'c'
                # result[1] -> index
                # result[2] -> patterns
                if result[0] == 'r':
                    # opp['r'][result[1]] = None
                    self.row_patterns[result[1]] = result[2]
                    self.sync_row(result[1])
                else:
                    # opp['c'][result[1]] = None
                    self.col_patterns[result[1]] = result[2]
                    self.sync_col(result[1])
                # print(opp)
            else:
                print()

    def sync_patterns_row(self, index):
        if not np.count_nonzero(self.coordinate[index]) == self.row:
            self.row_patterns[index] = self.consistent_patterns(
                self.row_patterns[index], self.coordinate[index])

    def sync_patterns_col(self, index):
        if not np.count_nonzero(self.coordinate[:, index]) == self.col:
            self.col_patterns[index] = self.consistent_patterns(
                self.col_patterns[index], self.coordinate[:, index])

    def sync_col(self, index):
        self.coordinate[:, index] = np.bitwise_or(
            self.coordinate[:, index], self.pattern_consensus(self.col_patterns[index]))

    def sync_row(self, index):
        self.coordinate[index] = np.bitwise_or(
            self.coordinate[index], self.pattern_consensus(self.row_patterns[index]))

    def pattern_consensus(self, pattern):
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


class NonogramHacker(Nonogram):
    def draw(self):
        if self.solved:
            base = np.where(self.coordinate == 1,
                            Draw.blue_block, Draw.white_block)
            for b in base:
                print(''.join(b) + Style.RESET_ALL)
        else:
            print('Nonogram is not solved yet...')

    def status(self):
        percentage = round((np.count_nonzero(self.coordinate) /
                            (self.row * self.col)) * 100, 1)
        print(f'[{self.row}X{self.col}] coordinate({percentage}%):')
        print(self.coordinate)

    def verify(self):
        return False if 0 in self.coordinate else True

    def sync_consensus(self):
        for i in range(self.row):
            self.sync_row(i)
        for i in range(self.col):
            self.sync_col(i)

    def sync_patterns(self):
        for i in range(self.row):
            self.sync_patterns_row(i)
        for i in range(self.col):
            self.sync_patterns_col(i)

    def solve(self):
        prev_nonzero_count = np.count_nonzero(self.coordinate)
        while not self.verify():
            self.sync_patterns()
            self.sync_consensus()
            nonzero_count = np.count_nonzero(self.coordinate)
            if prev_nonzero_count == nonzero_count:
                print("I can't solve this anymore...")
                self.status()
                break
            else:
                prev_nonzero_count = nonzero_count
        else:
            self.solved = True


def parse_from_text(keystext):
    def parse(keytext):
        return tuple(tuple(map(int, _.strip().split(' ')))
                     for _ in keytext.split(','))
    keytext = keystext.split('\n\n')
    rowkeytext = ''.join(keytext[0].split('\n'))
    colkeytext = ''.join(keytext[1].split('\n'))
    return parse(rowkeytext), parse(colkeytext)


def parse_from_file(filename):
    with open(filename) as f:
        return parse_from_text(f.read())
