from nonogram import *
from patterns import *
from multiprocessing import *
from time import time
from pprint import pprint
import sys
import itertools as it

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

def test(row_keys, col_keys, processes=None):
    nn = NonogramHacker(row_keys, col_keys, processes)
    start_time = time()
    nn.init_patterns()
    init_patterns_time = time()-start_time
    start_time = time()
    nn.solve()
    solving_time = time()-start_time
    nn.draw()
    return init_patterns_time, solving_time

def main(argv):
    if len(sys.argv) == 2:
        row_keys, col_keys = parse_from_file(sys.argv[1])
    else:
        # row_keys, col_keys = parse_from_file('test/55')
        # row_keys, col_keys = parse_from_file('test/1010')
        # row_keys, col_keys = parse_from_file('test/1515')
        # row_keys, col_keys = parse_from_file('test/2020')
        # row_keys, col_keys = parse_from_file('test/2525')
        row_keys, col_keys = parse_from_file('test/3030')

    test(row_keys, col_keys)

def test_performance():
    test_files = [
        'test/55',
        'test/1010',
        'test/1515',
        'test/2020',
        'test/2525',
        'test/3030'
    ]
    for test_file in test_files:
        row_keys, col_keys = parse_from_file(test_file)
        init_patterns_time, solving_time = test(row_keys, col_keys)
        floating_point = 4
        print(f'Time Taken(Init Pattern/Solving):{round(init_patterns_time, floating_point)}/{round(solving_time, 4)} secs')

def test_processes():
    # test_file = 'test/55'
    # test_file = 'test/1010'
    # test_file = 'test/1515'
    # test_file = 'test/2020'
    # test_file = 'test/2525'
    test_file = 'test/3030'
    row_keys, col_keys = parse_from_file(test_file)
    time_lst = []
    for proc_count in range(cpu_count(), 0, -1):
        print('Proc', proc_count)
        time_lst.append((proc_count, sum(test(row_keys, col_keys, proc_count))))
    print(time_lst)
    print('Best:', min(time_lst, key=lambda x:x[1]))
    print('Worst:', max(time_lst, key=lambda x:x[1]))

if __name__ == '__main__':
    # main(sys.argv)
    test_performance()
    # test_processes()

    # ll = 10**5
    # for i in range(ll):
    #     sys.stderr.write('\rdone {0:%}'.format((i+1)/ll))
    # sys.stderr.write('\n')