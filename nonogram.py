from itertools import permutations, chain
from multiprocessing import Pool, cpu_count
import numpy as np

DTYPE = np.dtype('i2')


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
            rchunk_divisions = cls._divisions(rchunk, space, lim=chunk, depth=depth+1)
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
            print(f'All pt:{sum(tuple(map(len, all_pt)))}')
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
        print(f'row/col:{self.row}/{self.col}')
        print('row_keys:', self.row_keys)
        print('col_keys:', self.col_keys)
        print(f'coordinate({np.count_nonzero(self.coordinate)}):')
        print(self.coordinate)

    def consensus(self, pt):
        thresh = pt.shape[0]
        def check_B(x): return 1 * (x == thresh)
        def check_W(x): return -1 * (x == -thresh)
        dist = np.sum(pt, axis=0)
        return check_B(dist) + check_W(dist)


def test(row_keys, col_keys, processes=None):
    nn = Nonogram(row_keys, col_keys)
    nn.status()
    for i in range(nn.row):
        nn.coordinate[i] = np.bitwise_or(
            nn.coordinate[i], nn.consensus(nn.row_pt[i]))
    nn.status()
    for i in range(nn.col):
        nn.coordinate[:, i] = np.bitwise_or(
            nn.coordinate[:, i], nn.consensus(nn.col_pt[i]))
    nn.status()

# Performance(Proc-16):0.6291401386260986
# Performance(Proc-15):0.6141374111175537
# Performance(Proc-14):0.5581254959106445
# Performance(Proc-13):0.5281186103820801
# Performance(Proc-12):0.5101144313812256
# Performance(Proc-11):0.4781064987182617
# Performance(Proc-10):0.4390990734100342
# Performance(Proc-9):0.41509294509887695
# Performance(Proc-8):0.3800852298736572
# Performance(Proc-7):0.3510775566101074
# Performance(Proc-6):0.3360757827758789
# Performance(Proc-5):0.3140695095062256
# Performance(Proc-4):0.30606889724731445
# Performance(Proc-3):0.29906702041625977
# Performance(Proc-2):0.27706217765808105
# ^.^ Performance(Proc-1):0.26805973052978516
sample55="""\
5;1 1 1;5;3;1 1 1
3 1;1 2;5;1 2;3 1
"""
# Performance(Proc-16):0.6381337642669678
# Performance(Proc-15):0.6081357002258301
# Performance(Proc-14):0.5561251640319824
# Performance(Proc-13):0.542121410369873
# Performance(Proc-12):0.5421221256256104
# Performance(Proc-11):0.5101137161254883
# Performance(Proc-10):0.45110106468200684
# Performance(Proc-9):0.4210946559906006
# Performance(Proc-8):0.38808655738830566
# Performance(Proc-7):0.369081974029541
# Performance(Proc-6):0.34607768058776855
# Performance(Proc-5):0.33007359504699707
# Performance(Proc-4):0.3200719356536865
# Performance(Proc-3):0.30606818199157715
# Performance(Proc-2):0.29706692695617676
# ^.^ Performance(Proc-1):0.29306507110595703
sample1010="""\
4;1 1;1 1 1 1;1 1 1 1;2 2;10;8;1 6 1;2 1 1 2;2 2 2
2 2;5 2;1 3 1;1 1 4;1 3 1;1 3 1;1 1 4;1 3 1;5 2;2 2
"""
# Performance(Proc-16):0.6841526031494141
# Performance(Proc-15):0.631141185760498
# Performance(Proc-14):0.6201391220092773
# Performance(Proc-13):0.5741288661956787
# Performance(Proc-12):0.5371203422546387
# Performance(Proc-11):0.4981114864349365
# Performance(Proc-10):0.5071136951446533
# Performance(Proc-9):0.4467349052429199
# Performance(Proc-8):0.4250950813293457
# Performance(Proc-7):0.40209007263183594
# Performance(Proc-6):0.36608123779296875
# Performance(Proc-5):0.35107970237731934
# Performance(Proc-4):0.33307433128356934
# ^.^ Performance(Proc-3):0.3200714588165283
# Performance(Proc-2):0.3390662670135498
# Performance(Proc-1):0.347078800201416
sample1515="""\
4 10;4 10;4 5;4 3 1 1;3 4 2;1 4 3;3 4;3 5;2 5;2 5;2 2 5;2 2 2;2 1 6;3 8;5
6;5 7;5 8;4 2 1;2;6 1;6 2;6 3;3 1 1;3 2;2 4 3;2 1 5 3;2 6 3;2 11;2 12
"""
# Performance(Proc-16):1.5764570236206055
# Performance(Proc-15):1.4748163223266602
# Performance(Proc-14):1.4693288803100586
# Performance(Proc-13):1.3573048114776611
# Performance(Proc-12):1.3365261554718018
# Performance(Proc-11):1.2887613773345947
# Performance(Proc-10):1.2592816352844238
# Performance(Proc-9):1.2542812824249268
# Performance(Proc-8):1.2512803077697754
# Performance(Proc-7):1.1427202224731445
# Performance(Proc-6):1.2552812099456787
# ^.^ Performance(Proc-5):1.223283290863037
# Performance(Proc-4):1.4582862854003906
# Performance(Proc-3):1.8644213676452637
# Performance(Proc-2):2.345524549484253
# Performance(Proc-1):3.9388837814331055
sample2020="""\
0;0;8;2 10 2;3 2 2 3;1 2 6 2 1;1 2 2 2 2 1;3 2 2 3;2 2 2 2;1 2 3 3 2 1;1 2 1 2 1 2 1;1 1 1 1 2 1 1 1 1;1 1 1 2 1 1 1;1 1 3 3 1 1;1 2 2 2 1;1 2 7 2 1;1 2 1 1 2 1;1 2 4 2 1;1 2 2 1;1 6 1
17;2 2;1 2 5;2 2 2;2 2 2;2 2 3 2;2 2 1 1 1 2;2 2 1 1 1 2 2;2 1 1 1 1 1 1;2 1 3 2 1 1;2 1 3 2 1 1;2 1 1 1 1 1 1;2 2 1 1 1 2 2;2 2 1 1 2;2 2 3 2;2 2 2;2 2 2;1 2 5;2 2;17
"""
# Performance(Proc-16):1.4663293361663818
# Performance(Proc-15):1.3462903499603271
# Performance(Proc-14):1.3112945556640625
# Performance(Proc-13):1.2432780265808105
# Performance(Proc-12):1.8571326732635498
# Performance(Proc-11):1.7310006618499756
# Performance(Proc-10):1.7654078006744385
# Performance(Proc-9):1.712714672088623
# Performance(Proc-8):1.6429665088653564
# Performance(Proc-7):1.5943572521209717
# Performance(Proc-6):1.3563034534454346
# ^.^ Performance(Proc-5):1.3553130626678467
# Performance(Proc-4):1.6173527240753174
# Performance(Proc-3):1.5323436260223389
# Performance(Proc-2):2.0635013580322266
# Performance(Proc-1):3.253593683242798
sample2525="""\
5 14;3 6 12;2 10 11;1 12 10;1 3 5 10;14 7;3 6 6;2 3 8 5;3 6 2 4;14 1 3;13 1 3;12 3;1 11 3;1 7 2 3;2 2 1 2 3;2 1 2 2 3;3 1 2 4;4 1 1 4;4 1 5;4 1 1 7;5 4 10;25;25;25;25
5 13;3 7 11;2 1 9 9;1 2 2 8 8;1 4 1 6 5;5 1 5 5 4;5 1 5 1 5;3 2 7 2 6;3 9 2 5;12 2 5;12 1 4;1 12 4;1 12 4;2 3 1 2 4;3 1 1 1 4;5 2 5;5 3 5;5 5;6 6;7 6;8 7;9 9;25;25;25
"""
# ^.^ Performance(Proc-16):18.457123279571533
# Performance(Proc-15):18.726192951202393
# Performance(Proc-14):19.43336057662964
# Performance(Proc-13):19.474360466003418
# Performance(Proc-12):19.941465139389038
# Performance(Proc-11):19.520370960235596
# Performance(Proc-10):19.752431631088257
# Performance(Proc-9):20.09549045562744
# Performance(Proc-8):20.080495595932007
# Performance(Proc-7):20.641621589660645
# Performance(Proc-6):22.151968479156494
# Performance(Proc-5):24.559513092041016
# Performance(Proc-4):26.13184118270874
# Performance(Proc-3):30.813899278640747
# Performance(Proc-2):36.566187381744385
# Performance(Proc-1):62.166932582855225
sample3030="""\
30;30;2 2;2 4 2;2 3 2;2 4 5 8 2;2 3 13 2 2;2 3 3 2;2 3 1 1 2;2 5 6 8 2;2 1 13 6 2;2 2;2 8 7 2;2 5 12 2;2 3 3 2;2 2 10 4 2;2 6 9 1 2;2 3 3 2;2 1 9 6 2;2 6 3 6 2;2 3 2;2 3 2 2;2 5 5 10 1 2;2 3 4 9 5 2;2 5 5 2 7 2;2 3 2;2 3 2;2 2 2;30;30
30;30;2 2;2 2 2;2 2 3 2;2 1 1 3 2;2 2 2 7 1 1 2;2 3 2 6 1 3 2;2 1 4 3 4 2 2;2 4 1 2 2 2 2 2;2 1 3 2 2 2 3 2;2 2 2 1 2 2 1 1 2;2 2 2 2 1 1 3 2;2 1 2 2 2 2 2 3 2;2 4 2 2 2 4 7;2 4 2 1 2 6 5;2 2 1 1 2 2 1 7 2;2 2 2 2 2 1 3 2;2 2 2 2 2 2 2 2;2 2 2 2 1 2 3 2;2 3 2 2 2 2 3 2;2 6 8 1 1 2;2 1 1 2 4 3 3 2;2 2 2 6 3 2;2 2 2 4 2;2 1 1 2 2;2 2 2;2 2;30;30
"""

def parse_from_text(keystxt):
    def parse(keytxt): return tuple(tuple(map(int, _.split(' ')))
                                    for _ in keytxt.split(';'))
    keytxt = keystxt.split('\n')
    return parse(keytxt[0]), parse(keytxt[1])


if __name__ == '__main__':
    from time import time
    # row_keys, col_keys = parse_from_text(sample55)
    # row_keys, col_keys = parse_from_text(sample1010)
    # row_keys, col_keys = parse_from_text(sample1515)
    row_keys, col_keys = parse_from_text(sample2020)
    # row_keys, col_keys = parse_from_text(sample2525)
    # row_keys, col_keys = parse_from_text(sample3030)
    test(row_keys, col_keys)
    # for i in range(cpu_count(), 0, -1):
    #     start = time()
    #     test(row_keys, col_keys, i)
    #     print(f'Performance(Proc-{i}):{time() - start}')

    '''
    permutation 알고리즘의 최적화 : 단순히 permutation 의 set 을 구하는 식으로 패턴을 구하고 있는데 이 방식은 중복 패턴을 계산하기에 속도가 느림 
    자료구조 단순화 : True, False 를 비트 단위의 1, 0 으로 대체하고 이 자료를 계산하는 방식도 bit 연산자로 교체 
    확정 요소 : 확정된 요소가 있을 때 그것을 기반으로 패턴을 생성하면 계산 시간이 단축됨 
    메모리 절약 : combinations 까지는 generator 라서 괜찮은데 patterns 부터 패턴을 저장하는 np.array 가 너무 커짐. 
        _divisions(list)->
        _division_with_filled_space(yield)->
        combinations(yield)->
        patterns(np.array)->
        pool.map(list)
    '''
