# http://egloos.zum.com/mononoct/v/262313
'''
?????
oxoox
oxxoo
xoxoo

키 값이 이미 두여있는지 확인한다. 
가장 처음 키값을 둔다. 
'''

# n 개의 자리. m 개의 공 

# 3 개의 자리. 2 개의 공 
'''
(oo) () ()
() (oo) ()
() () (oo)
(o) (o) ()
(o) () (o)
() (o) (o)
'''
# 3 개의 자리. 3 개의 공 
'''
(ooo) () ()
() (ooo) ()
() () (ooo)
(oo) (o) ()
(oo) () (o)
() (oo) (o)
(o) (oo) ()
(o) () (oo)
() (o) (oo)
가장 많이 뭉쳐있는 공이 m/2 보다 작아지면(이 경우 3/2) 
더 이상 의미 없기 때문에 중단. 
'''

'''
그러므로 m 이 짝수 일 때 가장 큰 공이 
m 개 일 경우, 
m-1 개 일 경우, 
...
(m/2)+1 개 일 경우, 
m/2 개 일 경우
로 나누어 생각할 수 있고 

m 이 홀수 일 때 가장 큰 공이 
m 개 일 경우, 
m-1 개 일 경우, 
...
(m/2)+3/2 개 일 경우, 
(m/2)+1/2 개 일 경우, 
로 나누어 생각할 수 있다. 

각각의 경우는 모두 서로 다르다는 것이 자명한데 
각각의 경우에서 가장 큰 공이 다르기 때문이다. 

이때 각각의 경우에서 가장 큰 공을 제외한 나머지 공들로
위와 같이 경우의 수를 나누어 공들이 배치되는 경우의 수를 구할 수 있다. 
'''

'''
m 개의 공을 n 개의 자리에 배치하는 경우의 수

1 개의 공을 1 개의 자리에 배치하는 경우의 수 == 1
(o)
1 개의 공을 2 개의 자리에 배치하는 경우의 수 == 2 
(o) ()
() (o)
1 개의 공을 3 개의 자리에 배치하는 경우의 수 == 3
(o) () ()
() (o) ()
() () (o)
1 개의 공을 n 개의 자리에 배치하는 경우의 수 == n

2 개의 공을 1 개의 자리에 배치하는 경우의 수 == 1 
(oo)
2 개의 공을 2 개의 자리에 배치하는 경우의 수 == 3 
(oo) ()
() (oo)
(o) (o)
2 개의 공을 3 개의 자리에 배치하는 경우의 수 == 6 
(oo) () ()
() (oo) ()
() () (oo)
(o) (o) ()
(o) () (o)
() (o) (o)
2 개의 공을 4 개의 자리에 배치하는 경우의 수 == 10 
(oo) () () ()
() (oo) () ()
() () (oo) ()
() () () (oo)
(o) (o) () ()
(o) () (o) ()
(o) () () (o)
() (o) (o) ()
() (o) () (o)
() () (o) (o)
2 개의 공을 5 개의 자리에 배치하는 경우의 수 == 15 
(oo) () () () ()
() (oo) () () ()
() () (oo) () ()
() () () (oo) ()
() () () () (oo)
(o) (o) () () ()
(o) () (o) () ()
(o) () () (o) ()
(o) () () () (o)
() (o) (o) () ()
() (o) () (o) ()
() (o) () () (o)
() () (o) (o) ()
() () (o) () (o)
() () () (o) (o)
2 개의 공을 n 개의 자리에 배치하는 경우의 수 == n(n+1)/2

3 개의 공을 1 개의 자리에 배치하는 경우의 수 == 1 
(ooo)
3 개의 공을 2 개의 자리에 배치하는 경우의 수 == 4
(ooo) ()
() (ooo)
(oo) (o)
(o) (oo)
3 개의 공을 3 개의 자리에 배치하는 경우의 수 == 9
(ooo) () ()
() (ooo) ()
() () (ooo)
(oo) (o) ()
(oo) () (o)
(o) (oo) ()
() (oo) (o)
(o) () (oo)
() (o) (oo)
3 개의 공을 4 개의 자리에 배치하는 경우의 수 == 16
(ooo) () () ()
() (ooo) () ()
() () (ooo) ()
() () () (ooo)
(oo) (o) () ()
(oo) () (o) ()
(oo) () () (o)
(o) (oo) () ()
() (oo) (o) ()
() (oo) () (o)
(o) () (oo) ()
() (o) (oo) ()
() () (oo) (o)
(o) () () (oo)
() (o) () (oo)
() () (o) (oo)
3 개의 공을 n 개의 자리에 배치하는 경우의 수 == n^2

4 개의 공을 1 개의 자리에 배치하는 경우의 수 == 1 
(oooo)
4 개의 공을 2 개의 자리에 배치하는 경우의 수 == 5
(oooo) ()
() (oooo)
(ooo) (o)
(o) (ooo)
(oo) (oo)
4 개의 공을 3 개의 자리에 배치하는 경우의 수 == 13
(oooo) () ()
() (oooo) ()
() () (oooo)
(ooo) (o) ()
(ooo) () (o)
(o) (ooo) ()
() (ooo) (o)
(o) () (ooo)
() (o) (ooo)
(oo) (oo) ()
(oo) () (oo)
(oo) (o) (o)
() (oo) (oo)
4 개의 공을 4 개의 자리에 배치하는 경우의 수 == 26
(oooo) () () ()
() (oooo) () ()
() () (oooo) ()
() () () (oooo)
(ooo) (o) () ()
(ooo) () (o) ()
(ooo) () () (o)
(o) (ooo) () ()
() (ooo) (o) ()
() (ooo) () (o)
(o) () (ooo) ()
() (o) (ooo) ()
() () (ooo) (o)
(o) () () (ooo)
() (o) () (ooo)
() () (o) (ooo)
(oo) (oo) () ()
(oo) () (oo) ()
(oo) () () (oo)
(oo) (o) (o) ()
(oo) (o) () (o)
(oo) () (o) (o)
() (oo) (oo) ()
() (oo) () (oo)
() (oo) (o) (o)
() () (oo) (oo)
4 개의 공을 n 개의 자리에 배치하는 경우의 수 == 
    (1/6)n^3 + n^2 -(1/6)n ==
    n(n^2 + 6n - 1) / 6 ==
    n(n + 3 - 10^(1/2))(n + 3 + 10^(1/2))/6
'''

def permutations_origin(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool) # length of iterable 
    r = n if r is None else r # length of permutation 
    if r > n:
        return
    indices = list(range(n)) # indices of permutation 
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def permutations_t(iterable):
    pool = tuple(iterable)
    n = len(pool) # length of iterable 
    indices = list(range(n)) # indices of permutation 
    cycles = list(range(n, 0, -1))
    yield tuple(pool[i] for i in indices[:n])
    while n:
        for i in reversed(range(n)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:n])
                break
        else:
            return

from itertools import permutations

def division(num, limit=float('inf'), root=True):
    for main_chunk in range(min(num, limit), 0, -1):
        rest_chunk = num - main_chunk
        rest_chunk_divisions = tuple(division(rest_chunk, limit=main_chunk, root=False))
        if rest_chunk_divisions:
            for rest_chunk_division in rest_chunk_divisions:
                yield [main_chunk, *rest_chunk_division]
        else: 
            yield [main_chunk]

def division_with_filled_space(space, num):
    for d in division(num):
        while len(d) != space:
            d.append(0)
        yield d

def test_division(space, n, m=None):
    if m == None: 
        m = n+1
    for i in range(n, m):
        print('='*20, 'division TEST for', i, '='*20)
        for v in division_with_filled_space(space, i):
            print(v)
    print('='*20, 'END  TEST', '='*20)

def tmp_permutations(space, n, m=None):
    if m == None: 
        m = n+1
    for i in range(n, m):
        # print('='*20, 'permutations TEST for', i, '='*20)
        for v in division_with_filled_space(space, i):
            yield set(permutations(v))
            # print(set(permutations(v)))
            # print(tuple(permutations(v)))
    # print('='*20, 'END  TEST', '='*20)

# test_division(8, 4)

def f(keys, L):
    S = len(keys)
    B = sum(keys)
    W = L - B
    print('S:', S, 'B:', B, 'W:', W, 'W - (S - 1):', W - (S - 1))
    rst = tuple(tmp_permutations(S+1, W - (S - 1)))
    print(rst)
    r = []
    if rst:
        for rs in rst:
            pass
    else:
        for i, key in enumerate(keys):
            for k in range(key):
                r.append(True)
            if i != len(keys) - 1:
                r.append(False)
    return r

# print(f((3, 1), 5))
f((1, 2), 5)

# print(set(p([3, 0, 0])))
# print(set(p([2, 1, 0])))
# print(set(p([1, 1, 1])))
