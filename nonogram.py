from itertools import permutations, chain
from multiprocessing import Pool
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
    def _divisions(cls, num, lim=float('inf')):
        for chunk in range(min(num, lim), 0, -1):
            rchunk = num - chunk
            rchunk_divisions = cls._has_next(
                cls._divisions(rchunk, lim=chunk))
            if rchunk_divisions is None:
                yield [chunk]
            else:
                for rest_chunk_division in rchunk_divisions:
                    yield [chunk, *rest_chunk_division]

    @classmethod
    def _division_with_filled_space(cls, space, num):
        for division in cls._divisions(num):
            while len(division) < space:
                division.append(0)
            yield division

    @classmethod
    def _combinations(cls, space, num):
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
            cls._combinations(S+1, W - (S - 1)))
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
    def __init__(self, row_keys, col_keys):
        self.row_keys = row_keys
        self.col_keys = col_keys
        self.row = len(self.row_keys)
        self.col = len(self.col_keys)
        self.coordinate = np.zeros((self.row, self.col), dtype=DTYPE)
        # self.row_pt = [Pattern.patterns(row_keys[i], self.row) for i in range(self.row)]
        # self.col_pt = [Pattern.patterns(col_keys[i], self.col) for i in range(self.col)]
        with Pool(processes=9) as pool:
            self.row_pt = pool.map(Pattern.patterns_from_map, [(row_keys[i], self.row) for i in range(self.row)])
            self.col_pt = pool.map(Pattern.patterns_from_map, [(col_keys[i], self.col) for i in range(self.col)])
    
    def status(self):
        print('row:', self.row)
        print('col:', self.col)
        print('row_keys:', self.row_keys)
        print('col_keys:', self.col_keys)
        print(f'coordinate({np.count_nonzero(self.coordinate)}):')
        print(self.coordinate)

    def consensus(self, pt):
        thresh = pt.shape[0]
        check_B = lambda x: 1 * (x == thresh)
        check_W = lambda x: -1 * (x == -thresh)
        dist = np.sum(pt, axis=0)
        return check_B(dist) + check_W(dist)

def test(row_keys, col_keys):
    nn = Nonogram(row_keys, col_keys)
    # nn.status()
    for i in range(nn.row):
        nn.coordinate[i] = np.bitwise_or(nn.coordinate[i], nn.consensus(nn.row_pt[i]))
    # nn.status()
    for i in range(nn.col):
        nn.coordinate[:, i] = np.bitwise_or(nn.coordinate[:, i], nn.consensus(nn.col_pt[i]))
    # nn.status()

if __name__ == '__main__':
    # http://egloos.zum.com/mononoct/v/262313
    # row_keys = (
    #     (5,), (1, 1, 1), (5,), (3,), (1, 1, 1)
    # )
    # col_keys = (
    #     (3, 1), (1, 2), (5,), (1, 2), (3, 1)
    # )
    row_keys = (
        (4,), (1,1), (1,1,1,1), (1,1,1,1), (2,2), (10,), (8,), (1,6,1), (2,1,1,2), (2,2,2)
    )
    col_keys = (
        (2,2), (5,2), (1,3,1), (1,1,4), (1,3,1), (1,3,1), (1,1,4), (1,3,1), (5,2), (2,2)
    )
    test(row_keys, col_keys)

    '''
    permutation 알고리즘의 최적화 : 단순히 permutation 의 set 을 구하는 식으로 패턴을 구하고 있는데 이 방식은 중복 패턴을 계산하기에 속도가 느림 
    자료구조 단순화 : True, False 를 비트 단위의 1, 0 으로 대체하고 이 자료를 계산하는 방식도 bit 연산자로 교체 
    확정 요소 : 확정된 요소가 있을 때 그것을 기반으로 패턴을 생성하면 계산 시간이 단축됨 
    멀티스레딩 / 멀티프로세싱 
    '''
