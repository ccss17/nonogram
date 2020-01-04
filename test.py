from nonogram import *

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

sample55 = """\
5;1 1 1;5;3;1 1 1
3 1;1 2;5;1 2;3 1
"""
sample1010 = """\
4;1 1;1 1 1 1;1 1 1 1;2 2;10;8;1 6 1;2 1 1 2;2 2 2
2 2;5 2;1 3 1;1 1 4;1 3 1;1 3 1;1 1 4;1 3 1;5 2;2 2
"""
sample1515 = """\
4 10;4 10;4 5;4 3 1 1;3 4 2;1 4 3;3 4;3 5;2 5;2 5;2 2 5;2 2 2;2 1 6;3 8;5
6;5 7;5 8;4 2 1;2;6 1;6 2;6 3;3 1 1;3 2;2 4 3;2 1 5 3;2 6 3;2 11;2 12
"""
sample2020 = """\
0;0;8;2 10 2;3 2 2 3;1 2 6 2 1;1 2 2 2 2 1;3 2 2 3;2 2 2 2;1 2 3 3 2 1;1 2 1 2 1 2 1;1 1 1 1 2 1 1 1 1;1 1 1 2 1 1 1;1 1 3 3 1 1;1 2 2 2 1;1 2 7 2 1;1 2 1 1 2 1;1 2 4 2 1;1 2 2 1;1 6 1
17;2 2;1 2 5;2 2 2;2 2 2;2 2 3 2;2 2 1 1 1 2;2 2 1 1 1 2 2;2 1 1 1 1 1 1;2 1 3 2 1 1;2 1 3 2 1 1;2 1 1 1 1 1 1;2 2 1 1 1 2 2;2 2 1 1 2;2 2 3 2;2 2 2;2 2 2;1 2 5;2 2;17
"""
sample2525 = """\
5 14;3 6 12;2 10 11;1 12 10;1 3 5 10;14 7;3 6 6;2 3 8 5;3 6 2 4;14 1 3;13 1 3;12 3;1 11 3;1 7 2 3;2 2 1 2 3;2 1 2 2 3;3 1 2 4;4 1 1 4;4 1 5;4 1 1 7;5 4 10;25;25;25;25
5 13;3 7 11;2 1 9 9;1 2 2 8 8;1 4 1 6 5;5 1 5 5 4;5 1 5 1 5;3 2 7 2 6;3 9 2 5;12 2 5;12 1 4;1 12 4;1 12 4;2 3 1 2 4;3 1 1 1 4;5 2 5;5 3 5;5 5;6 6;7 6;8 7;9 9;25;25;25
"""
sample3030 = """\
30;30;2 2;2 4 2;2 3 2;2 4 5 8 2;2 3 13 2 2;2 3 3 2;2 3 1 1 2;2 5 6 8 2;2 1 13 6 2;2 2;2 8 7 2;2 5 12 2;2 3 3 2;2 2 10 4 2;2 6 9 1 2;2 3 3 2;2 1 9 6 2;2 6 3 6 2;2 3 2;2 3 2 2;2 5 5 10 1 2;2 3 4 9 5 2;2 5 5 2 7 2;2 3 2;2 3 2;2 2 2;30;30
30;30;2 2;2 2 2;2 2 3 2;2 1 1 3 2;2 2 2 7 1 1 2;2 3 2 6 1 3 2;2 1 4 3 4 2 2;2 4 1 2 2 2 2 2;2 1 3 2 2 2 3 2;2 2 2 1 2 2 1 1 2;2 2 2 2 1 1 3 2;2 1 2 2 2 2 2 3 2;2 4 2 2 2 4 7;2 4 2 1 2 6 5;2 2 1 1 2 2 1 7 2;2 2 2 2 2 1 3 2;2 2 2 2 2 2 2 2;2 2 2 2 1 2 3 2;2 3 2 2 2 2 3 2;2 6 8 1 1 2;2 1 1 2 4 3 3 2;2 2 2 6 3 2;2 2 2 4 2;2 1 1 2 2;2 2 2;2 2;30;30
"""

def test(row_keys, col_keys, processes=None):
    import numpy as np
    nn = Nonogram(row_keys, col_keys)
    nn.status()
    nn.solve_row()
    nn.status()
    nn.discard()

if __name__ == '__main__':
    from time import time
    # row_keys, col_keys = parse_from_text(sample55)
    row_keys, col_keys = parse_from_text(sample1010)
    # row_keys, col_keys = parse_from_text(sample1515)
    # row_keys, col_keys = parse_from_text(sample2020)
    # row_keys, col_keys = parse_from_text(sample2525)
    # row_keys, col_keys = parse_from_text(sample3030)
    test(row_keys, col_keys)
