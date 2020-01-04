from nonogram import *

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

# def permutations_origin(iterable, r=None):
#     pool = tuple(iterable)
#     n = len(pool) # length of iterable 
#     r = n if r is None else r # length of permutation 
#     if r > n:
#         return
#     indices = list(range(n)) # indices of permutation 
#     cycles = list(range(n, n-r, -1))
#     yield tuple(pool[i] for i in indices[:r])
#     while n:
#         for i in reversed(range(r)):
#             cycles[i] -= 1
#             if cycles[i] == 0:
#                 indices[i:] = indices[i+1:] + indices[i:i+1]
#                 cycles[i] = n - i
#             else:
#                 j = cycles[i]
#                 indices[i], indices[-j] = indices[-j], indices[i]
#                 yield tuple(pool[i] for i in indices[:r])
#                 break
#         else:
#             return

"""
combination 에서 set(itertools.permutations) 을 사용해서 상당히 비효율적
    가장 많은 요소를 위치불변요소로 상정하고 남은 자리를 나머지 요소들 채워나가는 식으로 알고리즘 최적화. 이때 채워나감과 동시에 또 다른 자리가 생겨나는 식으로 경우의 수가 늘어남. 
permutation 알고리즘의 최적화
    단순히 permutation 의 set 을 구하는 식으로 패턴을 구하고 있는데 이 방식은 중복 패턴을 계산하기에 속도가 느림 
자료구조 단순화 : True, False 를 비트 단위의 1, 0 으로 대체하고 이 자료를 계산하는 방식도 bit 연산자로 교체 
확정 요소 : 확정된 요소가 있을 때 그것을 기반으로 패턴을 생성하면 계산 시간이 단축됨 
메모리 절약 : combinations 까지는 generator 라서 괜찮은데 patterns 부터 패턴을 저장하는 np.array 가 너무 커짐. 
    _divisions(list)->
    _division_with_filled_space(yield)->
    combinations(yield)->
    patterns(np.array)->
    pool.map(list)
Patterns 클래스에서도 순수하게 numpy.array 사용 
    _divisions -> squeeze 함수 
    _divisions_with_filled_space -> broadcast or 확장관련함수 
    
"""

def test(row_keys, col_keys, processes=None):
    nn = Nonogram(row_keys, col_keys)
    nn.solve()
    nn.draw()

if __name__ == '__main__':
    from time import time
    import sys
    row_keys, col_keys = parse_from_file(sys.argv[1])
    start_time = time()
    test(row_keys, col_keys)
    print(Style.RESET_ALL+f'Time taken:{round(time()-start_time, 1)} secs')
