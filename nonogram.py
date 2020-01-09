from multiprocessing import Pool, cpu_count
import numpy as np
from colorama import Fore, Style, init
from patterns import *
from config import *
import sys
from pprint import pprint

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
        time_list = []

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
                # print()
                # result[0] -> 'r' or 'c'
                # result[1] -> index
                # result[2] -> patterns
                if result[0] == 'r':
                    # opp['r'][result[1]] = None
                    self.row_patterns[result[1]] = result[2]
                    self.sync_coordinate_row(result[1])
                else:
                    # opp['c'][result[1]] = None
                    self.col_patterns[result[1]] = result[2]
                    self.sync_coordinate_col(result[1])
                # print(opp)
                time_list.append((result[0], result[1], result[3]))
                # print(self.coordinate)
            else:
                print()
            time_list.sort(key=lambda x:x[2][0])
            # pprint(time_list)
    '''
    Sample 1
    (0.0, 21, 1, 0.047619047619047616),
    (0.0, 12, 1, 0.08333333333333333),
    (0.0, 7, 1, 0.14285714285714285),
    (0.0, 23, 1, 0.043478260869565216),
    (0.0, 23, 2, 0.08695652173913043),
    (0.0, 26, 1, 0.038461538461538464),
    (0.0010008811950683594, 12, 1, 0.08333333333333333),
    (0.0020008087158203125, 19, 2, 0.10526315789473684),
    (0.0029997825622558594, 21, 3, 0.14285714285714285),
    (0.003000974655151367, 20, 3, 0.15),
    (0.003001689910888672, 21, 3, 0.14285714285714285),
    (0.00400090217590332, 15, 2, 0.13333333333333333),
    (0.004001617431640625, 20, 3, 0.15),
    (0.004002809524536133, 10, 2, 0.2),
    (0.0050008296966552734, 9, 2, 0.2222222222222222),
    (0.0050008296966552734, 9, 2, 0.2222222222222222),
    (0.0050013065338134766, 11, 2, 0.18181818181818182),
    (0.005998849868774414, 8, 2, 0.25),
    (0.005998849868774414, 7, 2, 0.2857142857142857),
    (0.006001949310302734, 6, 2, 0.3333333333333333),
    (0.0060040950775146484, 7, 2, 0.2857142857142857),
    (0.007001399993896484, 18, 3, 0.16666666666666666),
    (0.007001399993896484, 4, 2, 0.5),
    (0.007001638412475586, 6, 2, 0.3333333333333333),
    (0.007001638412475586, 6, 2, 0.3333333333333333),
    (0.007001638412475586, 4, 2, 0.5),
    (0.00800180435180664, 17, 3, 0.17647058823529413),
    (0.008002281188964844, 4, 2, 0.5),
    (0.012002229690551758, 16, 3, 0.1875),
    (0.01500248908996582, 14, 3, 0.21428571428571427),
    (0.017002582550048828, 14, 3, 0.21428571428571427),
    (0.020003318786621094, 12, 3, 0.25),
    (0.02100396156311035, 12, 3, 0.25),
    (0.022004127502441406, 12, 3, 0.25),
    (0.033007144927978516, 9, 3, 0.3333333333333333),
    (0.03500699996948242, 16, 4, 0.25),
    (0.04500842094421387, 8, 3, 0.375),
    (0.04700970649719238, 7, 3, 0.42857142857142855),
    (0.047011375427246094, 7, 3, 0.42857142857142855),
    (0.06601381301879883, 14, 4, 0.2857142857142857),
    (0.07601714134216309, 13, 4, 0.3076923076923077),
    (0.08001923561096191, 13, 4, 0.3076923076923077),
    (0.08101582527160645, 13, 4, 0.3076923076923077),
    (0.08201861381530762, 13, 4, 0.3076923076923077),
    (0.09001922607421875, 17, 6, 0.35294117647058826),
    (0.12502765655517578, 15, 5, 0.3333333333333333),
    (0.1290292739868164, 11, 4, 0.36363636363636365),
    (0.26806092262268066, 13, 5, 0.38461538461538464),
    (0.27006077766418457, 13, 5, 0.38461538461538464),
    (0.2740621566772461, 13, 5, 0.38461538461538464),
    (0.27706336975097656, 13, 5, 0.38461538461538464),
    (0.38408613204956055, 12, 5, 0.4166666666666667),
    (0.4360971450805664, 14, 6, 0.42857142857142855),
    (0.6291415691375732, 11, 5, 0.45454545454545453),
    (0.8151841163635254, 11, 5, 0.45454545454545453),
    (0.9792201519012451, 9, 5, 0.5555555555555556),
    (1.117250919342041, 10, 5, 0.5),
    (1.485358476638794, 10, 5, 0.5),
    (92.25744032859802, 10, 7, 0.7),
    (107.80640006065369, 11, 8, 0.7272727272727273)
    '''

    '''
    Sample 2
    (0.0, 26, 3, 0.11538461538461539),
    (0.0, 25, 3, 0.12),
    (0.0010004043579101562, 24, 4, 0.16666666666666666),
    (0.0019996166229248047, 20, 2, 0.1),
    (0.0020012855529785156, 23, 4, 0.17391304347826086),
    (0.0030019283294677734, 16, 2, 0.125),
    (0.004000425338745117, 20, 3, 0.15),
    (0.004000425338745117, 20, 3, 0.15),
    (0.004001617431640625, 20, 3, 0.15),
    (0.005000114440917969, 21, 4, 0.19047619047619047),
    (0.006001949310302734, 21, 6, 0.2857142857142857),
    (0.007001399993896484, 21, 5, 0.23809523809523808),
    (0.00800323486328125, 20, 4, 0.2),
    (0.010001897811889648, 16, 3, 0.1875),
    (0.012001752853393555, 19, 4, 0.21052631578947367),
    (0.012002706527709961, 20, 5, 0.25),
    (0.012002944946289062, 20, 5, 0.25),
    (0.012002944946289062, 15, 3, 0.2),
    (0.013002157211303711, 20, 5, 0.25),
    (0.01600337028503418, 14, 3, 0.21428571428571427),
    (0.017003297805786133, 14, 3, 0.21428571428571427),
    (0.01800370216369629, 13, 3, 0.23076923076923078),
    (0.021004199981689453, 12, 3, 0.25),
    (0.022005558013916016, 19, 5, 0.2631578947368421),
    (0.023003816604614258, 12, 3, 0.25),
    (0.024004697799682617, 11, 3, 0.2727272727272727),
    (0.02700519561767578, 11, 3, 0.2727272727272727),
    (0.027005910873413086, 11, 3, 0.2727272727272727),
    (0.028007030487060547, 17, 4, 0.23529411764705882),
    (0.031008005142211914, 10, 3, 0.3),
    (0.03300786018371582, 16, 4, 0.25),
    (0.035007476806640625, 18, 5, 0.2777777777777778),
    (0.03800797462463379, 16, 4, 0.25),
    (0.0400090217590332, 15, 4, 0.26666666666666666),
    (0.04201078414916992, 8, 3, 0.375),
    (0.04401063919067383, 8, 3, 0.375),
    (0.046010494232177734, 15, 4, 0.26666666666666666),
    (0.04901003837585449, 18, 6, 0.3333333333333333),
    (0.05501270294189453, 17, 5, 0.29411764705882354),
    (0.060011863708496094, 14, 4, 0.2857142857142857),
    (0.06201362609863281, 14, 4, 0.2857142857142857),
    (0.07901716232299805, 13, 4, 0.3076923076923077),
    (0.07901811599731445, 13, 4, 0.3076923076923077),
    (0.08301830291748047, 16, 5, 0.3125),
    (0.08301901817321777, 16, 5, 0.3125),
    (0.1000223159790039, 15, 5, 0.3333333333333333),
    (0.10102295875549316, 12, 4, 0.3333333333333333),
    (0.1270284652709961, 15, 5, 0.3333333333333333),
    (0.13302946090698242, 11, 4, 0.36363636363636365),
    (0.13503003120422363, 11, 4, 0.36363636363636365),
    (0.14003252983093262, 11, 4, 0.36363636363636365),
    (0.14603233337402344, 10, 4, 0.4),
    (0.25905799865722656, 8, 4, 0.5),
    (0.2670600414276123, 13, 5, 0.38461538461538464),
    (0.27706170082092285, 13, 5, 0.38461538461538464),
    (0.3270730972290039, 12, 5, 0.4166666666666667),
    (0.3780844211578369, 12, 5, 0.4166666666666667),
    (0.48612141609191895, 11, 5, 0.45454545454545453),
    (0.6021833419799805, 13, 6, 0.46153846153846156),
    (0.9292669296264648, 12, 6, 0.5)
    '''

    def consistent_patterns(self, pattern_set, line):
        nonzero_indices = np.nonzero(line)[0]
        confirmation = line[nonzero_indices]
        consistency = []
        for i, pattern in enumerate(pattern_set):
            if np.array_equal(confirmation, pattern[nonzero_indices]):
                consistency.append(i)
        return pattern_set[consistency]

    def pattern_consensus(self, pattern):
        thresh = pattern.shape[0]
        def check_B(x): return 1 * (x == thresh)
        def check_W(x): return -1 * (x == -thresh)
        dist = np.sum(pattern, axis=0)
        return check_B(dist) + check_W(dist)

    def sync_patterns_row(self, index):
        if not np.count_nonzero(self.coordinate[index]) == self.row:
            self.row_patterns[index] = self.consistent_patterns(
                self.row_patterns[index], self.coordinate[index])

    def sync_patterns_col(self, index):
        if not np.count_nonzero(self.coordinate[:, index]) == self.col:
            self.col_patterns[index] = self.consistent_patterns(
                self.col_patterns[index], self.coordinate[:, index])

    def sync_coordinate_row(self, index):
        self.coordinate[index] = np.bitwise_or(
            self.coordinate[index], self.pattern_consensus(self.row_patterns[index]))

    def sync_coordinate_col(self, index):
        self.coordinate[:, index] = np.bitwise_or(
            self.coordinate[:, index], self.pattern_consensus(self.col_patterns[index]))


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
            self.sync_coordinate_row(i)
        for i in range(self.col):
            self.sync_coordinate_col(i)

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
