# http://egloos.zum.com/mononoct/v/262313
from itertools import permutations, chain
import numpy as np

class Nono:
    def __init__(self, row, col, keys):
        self.coordinate = np.zeros((row, col))
        # self.coordinate = np.arange(25).reshape(5, 5)
        self.keys = keys 

    def has_next(self, iterable):
        try:
            first = next(iterable)
        except StopIteration:
            return None
        return chain([first], iterable)

    def divisions(self, num, limit=float('inf'), root=True):
        for main_chunk in range(min(num, limit), 0, -1):
            rest_chunk = num - main_chunk
            rest_chunk_divisions = self.has_next(self.divisions(rest_chunk, limit=main_chunk, root=False))
            if rest_chunk_divisions is None:
                yield [main_chunk]
            else:
                for rest_chunk_division in rest_chunk_divisions:
                    yield [main_chunk, *rest_chunk_division]

    def division_with_filled_space(self, space, num):
        for division in self.divisions(num):
            while len(division) < space:
                division.append(0)
            yield division

    def tmp_permutations(self, space, n, m=None):
        if m == None: 
            m = n+1
        for i in range(n, m):
            for v in self.division_with_filled_space(space, i):
                for _patterns in set(permutations(v)):
                    yield _patterns

    def patterns(self, keys, length):
        S = len(keys)
        B = sum(keys)
        W = length - B
        _white_block_patterns = self.has_next(self.tmp_permutations(S+1, W - (S - 1)))
        # arr = np.array([[-1, -1, -1]], dtype=np.dtype('i2'))
        arr = np.empty((length), dtype=np.dtype('i2'))
        arr.fill(-1)
        arr = arr.reshape(1, length)
        if _white_block_patterns is None:
            _pattern = np.array([], dtype=np.dtype('i2'))
            for i, key in enumerate(keys):
                _pattern = np.append(_pattern, [1 for _ in range(key)])
                if i != len(keys) - 1:
                    _pattern = np.append(_pattern, 0)
            arr = np.array([_pattern], dtype=np.dtype('i2'))
        else:
            flag = True
            for _white_block in _white_block_patterns:
                _pattern = np.array([], dtype=np.dtype('i2'))
                for i, _white_block_size in enumerate(_white_block):
                    _pattern = np.append(_pattern, np.array([0 for _ in range(_white_block_size)], dtype=np.dtype('i2')))
                    if i < len(keys):
                        _pattern = np.append(_pattern, np.array([1 for _ in range(keys[i])], dtype=np.dtype('i2')))
                    if i < len(keys):
                        if i != len(keys) - 1:
                            _pattern = np.append(_pattern, 0)
                if arr[0, 0] == -1:
                    arr = np.array([_pattern], dtype=np.dtype('i2'))
                else:
                    arr = np.append(arr, np.array([_pattern], dtype=np.dtype('i2')), axis=0)
        return arr

    def np_consensus(self, arr):
        test = lambda x: 1 * (x == True)
        # top_thres = lambda x: 1 * (x == True)
        # bottom_thres = lambda x: 0 * (x == True)
        thres = arr.shape[0]
        dist = np.sum(arr, axis=0)
        print(dist)
        print(test(dist == thres))
        print(test(dist == 0))
        # top = top_thres(arr == thres)
        # bottom = bottom_thres(arr == 0)

    def consensus(self, iters):
        _consensus = []
        _prev = []
        for iter in iters:
            if not _consensus: 
                _consensus = iter
                _prev = iter
                continue
            for i, (a, b) in enumerate(zip(_prev, iter)):
                if _consensus is not None and a != b:
                    _consensus[i] = None
        return _consensus

    def test(self, keys, space):
        for p in self.patterns(keys, space):
            print("P:", p)
        print("C:", self.consensus(self.patterns(keys, space)))

keys = (
    (
        (5,), (1, 1, 1), (5,), (3,), (1, 1, 1)
    ),
    (
        (3, 1), (1, 2), (5,), (1, 2), (3, 1)
    )
)

nn = Nono(5, 5, keys)
# nn.test((5,), 5)
# nn.test((2, 1), 5)
# nn.test((2, 2), 5)
# nn.test((2, 2), 6)
# nn.test((3, 2, 1), 10)

print(nn.patterns((3, 2, 3), 11))
print(nn.np_consensus(nn.patterns((3, 2, 3), 11)))
# print(nn.patterns((3, 2, 1), 10).shape)
# print(np.sum(nn.patterns((3, 2, 1), 10), axis=0))
# print(np.sum(nn.patterns((3, 2, 1), 10), axis=0) == 10)
# print(np.sum(nn.patterns((3, 2, 1), 10), axis=0) == 0)

# print(nn.patterns((2, 2), 5))
# print(nn.patterns((2, 2), 5).shape)
# print(np.sum(nn.patterns((2, 2), 5), axis=0) )
# print(np.sum(nn.patterns((2, 2), 5), axis=0) == 1)
# print(np.sum(nn.patterns((2, 2), 5), axis=0) == 0)

# for pattern in nn.patterns((3, 2, 1), 10):
#     print(pattern)

# print(coordinate)
# print(coordinate[0])
# print(coordinate[:, 0]) # 

'''
permutation 알고리즘의 최적화 : 단순히 permutation 의 set 을 구하는 식으로 패턴을 구하고 있는데 이 방식은 중복 패턴을 계산하기에 속도가 느림 
자료구조 단순화 : True, False 를 비트 단위의 1, 0 으로 대체하고 이 자료를 계산하는 방식도 bit 연산자로 교체 
확정 요소 : 확정된 요소가 있을 때 그것을 기반으로 패턴을 생성하면 계산 시간이 단축됨 
'''