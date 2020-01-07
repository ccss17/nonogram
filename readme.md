# nonogram

[What is nonogram?](https://en.wikipedia.org/wiki/Nonogram)

[네모로직이란?](https://namu.wiki/w/%EB%85%B8%EB%85%B8%EA%B7%B8%EB%9E%A8https://namu.wiki/w/%EB%85%B8%EB%85%B8%EA%B7%B8%EB%9E%A8)

## Usage

```shell
# try sample like this
python main.py test/55
python main.py test/1010
python main.py test/1515
python main.py test/2020
python main.py test/2525
python main.py test/3030
```

![preivew](preview.gif)

![preivew](preview2.gif)

1. 네모로직의 가로키 세로키를 입력 

    - 파일에 입력하는 방식 

    - 스크립트에 입력하는 방식 

## TODO

- ㅇㅇ 

- 패턴을 만듦과 동시에 풀기 시작하기

  - 현재는 패턴생성 -> 풀기 이렇게 이산적으로 하고 있는데 다이나믹하게 패턴을 만들면서 풀기 

- Pool.map -> Pool.imap or async... (메모리 효율 + 생성자에서 stuck)

- 자료구조 단순화 : True, False 를 비트 단위의 1, 0 으로 대체하고 이 자료를 계산하는 방식도 bit 연산자로 교체 

- 메모리 절약 : combinations 까지는 generator 라서 괜찮은데 patterns 부터 패턴을 저장하는 np.array 가 너무 커짐. 

  - _divisions(list)->

  - _division_with_filled_space(yield)->

  - combinations(yield)->

  - patterns(np.array)->

  - pool.map(list)

- C 언어 사용 : combination 까지는 privimitve 타입을 ㅏㅅ용하기 때문에 C 언어로 대체하여 속도 올릴 수 있지 않을까 

  - _divisions(primitive)->

  - _division_with_filled_space(primitive)->

  - combinations(primitive)->

  - patterns(numpy)->

  - pool.map(numpy)

- Patterns 클래스에서도 순수하게 numpy.array 사용 

  - _divisions -> squeeze 함수 

  - _divisions_with_filled_space -> broadcast or 확장관련함수 

## Philosophy

### Idea

#### Definition 

- **패턴(Pattern)** : 키 값으로 정해지는 라인의 모양

  - 예시 : 키 값이 `"2 1"` 이고 라인의 길이가 `5` 일때 정해질 수 있는 패턴 중 하나는 `"ooxox"` 이다. 

- **패턴의 집합(Set of pattern)** : 패턴의 모든 경우의 집합 

  - 예시 : 키 값이 `"2 1"` 이고 라인의 길이가 `5` 일때 패턴은 집합 `{"ooxox", "ooxxo", "xooxo"}` 이다. 

- **패턴의 복잡도(Complexity of pattern)** : 패턴이 지니는 원소의 개수 

  - 예시 : 위의 예시에서 패턴의 복잡도는 `3` 이다. 

- **패턴의 교집합(Consensus of pattern)** : 패턴이 지니는 모든 원소들이 공통적으로 갖고 있는 동일한 블록들의 위치정보와 상태정보를 가진 집합 

  - 예시 : 위의 예시에서 패턴의 교집합은 `"?o???"` 이다. 

- **패턴의 축소(Reduction of pattern)** : 결정된 블록에 근거하여 존재하지 않는 패턴을 을 삭제하는 것.  

  - 예시 : 위의 예시에서 결정된 블록이 `"???x?"` 이라 할 때 패턴은 집합 `{"ooxxo", "xooxo"}` 으로 축소된다. 

- **패턴의 교집합의 확장(Expansion of consensus of pattern)** : 패턴의 축소로 인하여 패턴이 공통적으로 갖는 동일한 블록들이 늘어난 것 

  - 예시 : 위의 예시에서 결정된 블록이 `"???x?"` 이라 할 때 패턴은 집합 `{"ooxxo", "xooxo"}` 으로 축소되었다. 이때 패턴의 교집합은 `"?o???"` 에서 `"?o?xo"` 으로 확장된다. 

#### Solution

- (1) **네모로직의 생성** : 키 값으로 패턴이 결정된다. 

  - 그런데 패턴은 여러 형태를 가질 수 있기 때문에 라인의 블록들이 한번에 결정되지 않는다.

  - 만약 모든 패턴 형태가 동의하는 교집합이 존재하지 않으면 네모로직을 풀 수 없다.(증명 필요)
  
  - 하지만 풀 수 있는 네모로직은 모든 패턴 형태가 동의하는 교집합이 존재한다. 

- (2) **패턴의 교집합 찾기** : 모든 패턴 형태가 동의하는 교집합으로 라인의 일부 블록을 결정할 수 있다. 

- (3) **다른 라인의 패턴 축소** : 블록이 결정되면 그 블록을 포함하는 다른 라인의 패턴이 축소되고 이에 따라 해당 라인의 패턴의 교집합이 확장된다. 

- (4) **2 와 3의 반복** : 다른 라인의 교집합이 확장되었다면 다시 라인의 일부 블록을 결정할 수 있고 결정된 블록을 포함하는 다른 라인의 패턴이 축소된다. 이에 따라 또 다시 해당 라인의 패턴의 교집합이 확장된다. 

- (5) 이때 다음 반복로직은 패턴의 복잡도가 `1` 이 될 때 끝나며 그 시점에서 네모로직은 풀려있게 된다. 

  - 반복로직 : **(패턴의 교집합 찾기)** &rarr; **(다른 라인의 패턴 축소)** &rarr; **(패턴의 교집합 찾기)** &rarr; **(다른 라인의 패턴 축소)** &rarr; ... 

#### Further discussion

- 패턴의 복잡도가 패턴의 교집합에 비례하는가?

  - **배경** : 패턴의 교집합이 가장 큰 것부터 풀이를 시작해나가야 한다. 그런데 여러 패턴의 교집합이 발생하였을 때 어느 패턴부터 시작해야 풀이가 가장 간단해질까? 그것은 패턴의 교집합이 가장 큰 것부터 시작하는 것이다. 왜냐하면 교집합이 클 수록 영향을 받아 패턴의 축소가 발생하는 라인의 개수가 많아지기 때문이다. 이에따라 패턴의 교집합이 작은 것부터 풀이를 시작하는 것보다 전체적인 패턴의 복잡도의 총량이 더 낮아진다. 

  - **설명** : 그러므로 패턴의 교집합이 가장 큰 것을 비교하여 구해야 한다. 그런데 만약 패턴의 복잡도가 패턴의 교집합과 비례한다면 모든 패턴의 교집합까지 구할 필요 없이 패턴의 복잡도까지만 구한 시점에서 복잡도로 비교를 할 수 있게 된다. 이렇게 하면 절차가 한 단계 짧아지는 효과가 발생하여 풀이가 끝나는 전체 시간도 짧아진다. 

  - **반례** : 만약 패턴의 복잡도가 작아도 패턴의 교집합이 더 작을 수도 있다고 가정하자. 예를 들어 `n` 번째 라인의 패턴의 복잡도가 `2` 이고 교집합의 개수가 `1` 이라고 생각하자. 또 `m` 번째 라인의 패턴의 복잡도가 `3` 이고 교집합의 개수가 `2` 라고 가정해보자. 이 가정이 불가능하지 않다면 패턴의 복잡도는 패턴의 교집합의 개수와 비례하지 않는다. 

  - **반례의 증명** : 라인 `n` 은 패턴의 복잡도가 `2` 이고 교집합의 개수는 `1` 이다. 그러므로 패턴 `{"oooox", "xxxxx"}` 을 갖다고 생각하자. 라인 `m` 은 패턴의 복잡도가 `3` 이고 교집합의 개수는 `2` 이다. 그러므로 패턴 `{"oooxx", "xxxxx", "oxoxx"}` 갖는다고 생각하자. 그러면 라인 `n` 은 `"????x"` 라는 교집합을 갖고 라인 `m` 은 `"???xx"` 라는 교집합을 갖는다. 

  - **결론** : 라인의 패턴의 복잡도가 낮다고 해서 교집합의 개수가 반드시 많은 것은 아니다. (그러므로 항상 패턴의 교집합까지 구해서 비교해야 한다.)

- 만약 패턴의 교집합이 존재하지 않는다면 패턴의 집합에서 임의로 패턴 몇 개를 제거하여 패턴의 교집합을 강제로 축소하여 네모로직의 실질적 풀이를 진행시킬 수 있나?

  - 패턴의 교집합이 존재하지 않는다면 강제로 임의의 패턴 몇 개를 제거하여 풀이를 진행해야 하고 이때 네모로직의 전체 모양이 여러 경우의 수로 나뉠 수 있다. 

- 여러 패턴의 교집합이 발생하였을 때 먼저 처리해야 하는 패턴

  - 만약 여러 패턴에 교집합이 발생하였 

- 키 값으로 패턴이 어떻게 결정되는가?

  - ~~**단순화에 관하여** : 문제나 상황, 현상을 할 수 있는데까지 최대한 단순화시키는 것은 매우 중요한데 왜냐하면 불변요소와 가변요소를 결정한 시점에서 불변요소를 사고대상영역에서 제외시킬 수 있기 때문이다. 이렇게 되면 불변요소가 사라져서 좀 더 가벼워지고 단순해진 사고대상영역을 볼 수 있다. 그러면 생각을 진행시키기 훨씬 수월해진다. 또한 비록 문제와 상황의 논리가 허용하는 단순화가 아니라 할지라도 실험자가 임의로 가변요소를 불변요소로 정하여 강제로 단순화를 시키고 사고를 진행할 수도 있다. 이렇게 강제 단순화로 이끌어낸 진리에 대하여서는 "어떤 조건 하에서는 참이다" 또는 "어떤 조건 하에서 적용된다" 라고 하면 되기 때문이다.~~

  - **키 값** : 네모로직에서 키 값이란 수열이다. 이 수열은 일정한 개수의 빈 칸이 그려저 있는 테이프에 적용된다. 수열의 크기만큼 검은색 블록들이 존재하게 된다. 그 블록들은 수열의 숫자만큼의 검은색 칸의 나열로 이루어진다.  검은색 블록 사이에는 반드시 빈칸이 하나 이상 있어서 검은색 블록들을 분리한다. 

  - 빈 칸을 하얀색 칸이라고 정의하자. 그러면 검은색 칸 사이에 반드시 하얀색 칸이 하나 이상 있어야 한다. 

  - **가변요소** : 수열을 `keys` 라 하고, 수열의 크기(검은색 블록의 개수)가 `S` 개, 테이프의 칸의 개수가 `L` 개, 검은색 칸의 총합을 `B`, 하얀색 칸의 총합이 `W` 이라고 하자. 그리고 `i` 번째 검은색 블록을 전달하면 블록을 구성하는 칸의 개수를 반환해주는 함수 `f` 가 있다고 하자. 즉 `f(i)` 는 `i` 번째 검은색 블록의 칸개수이다. 그러면 다음 식이 성립한다. 

    `B = f(1) + f(2) + ... + f(S) = sum(keys)`

    `S = len(keys)`

    `W = L - B`

    `L = B + W = {f(1) + f(2) + ... + f(S)} + W = sum(keys) + W`

    그런데 키 값에 의해 `S` 개의 검은색 블록들은 이미 정해져 있다. 또 검은색 블록들 사이에 하얀색 칸이 반드시 `1` 개 이상 있다. 그러면 `i` 번째 검은색 블록과 그것에 붙어있는 하얀색 칸을 `f(i)+1` 라 할 수 있다. 또 이 검은색 블록의 사이와 앞뒤에 하얀색 칸이 임의로 배정된 다음의 수식을 생각할 수 있다. `x_i (i:1 ~ S+1)` 에 하얀색 칸이 `0` 개 이상 존재한다. 

    `L = x_1 + {f(1)+1} + x_2 + {f(2)+1} + x_3 + ... + x_(S-1) + {f(S-1)+1} + x_S + {f(S)} + x_(S+1)`

    `L = x_1 + x_2 + x_3 + ... + x_(S-1) + x_S + x_(S+1) + B + (S - 1)`

    `L - B - (S - 1) = W - (S - 1) = x_1 + x_2 + x_3 + ... + x_(S-1) + x_S + x_(S+1)`

    그러면 **키 값으로 결정되는 패턴**은 `S + 1` 개의 자리에 `W - (S - 1)` 개의 하얀색 칸을 어떻게 배정해야 하는지의 문제로 단순화된다. 

##### `n` 개의 공으로 이루어진 공을 나눠서 `m` 개의 자리에 둘 수 있는 모든 경우의 수 

- 이 문제는 `2` 가지 문제로 단순화된다. `1` 번째는 `n` 개의 공으로 이루어진 공을 나누는 것, `2` 번째는 다양한 크기의 공들을 `m` 개의 자리에 두는 것이다. 

###### `n` 개의 공으로 이루어진 공을 나누는 방법 

- `n` 개의 공으로 이루어진 공을 나눈 모양들은 가장 응집도가 높은 형태와 가장 응집도가 낮은 형태의 스펙트럼 사이에 존재한다. 가장 응집도가 높은 형태는 `n` 개의 공이 전부 다 뭉쳐져 있어 `1` 개의 자리를 차지하고 있는 형태이다. 가장 응집도가 낮은 형태는 모든 공이 흩어져 있어서 `n` 개의 자리를 차지하고 있는 형태이다. 

- 그러므로 모든 경우의 수는 가장 큰 덩어리가 `n` 인 형태, 가장 큰 덩어리가 `n-1` 인 형태, `...`, 가장 큰 덩어리가 `1` 인 형태로 구성된다. 

- ~~이 문제를 해결하는 알고리즘을 구상하고 있었는데 노트가 필요하다는 것을 느꼈다. 노트에 생각을 구체화시켜 기록해야 다음 생각을 발전시킬 수 있었다. 사실 구체화시키기도 전에 이미 생각이 존재한다는 것이 신기하긴 했었지만 어쩔 수 없다. 순서를 없애고 작은 원들을 구름 형태로 생각하면서 경우의 수를 나눠보니까 풀리기 시작했다. 순서에 대한 고정관념 때문에 알고리즘이 세워지는 게 계속 막히는 것 같아서 그렇게 했다.~~

### Policy

#### Definition 

  - 키 값의 개수 = `len(k)`

  - 키 값 = `k`

  - 첫번째 키 값 = `k[0]`

  - 마지막 키 값 = `k[len(k)-1]`

  - 키 값의 합 = `sum(k)`

  - 칸 수 = `n`

  - 이미 칠해진 칸 수 = `m`

  - 칸 전체를 O 으로 칠하기 = `line_on()`

  - 칸 전체를 X 으로 칠하기 = `line_off()`

  - 칠해지지 않은 칸을 O 으로 칠하기 = `remain_on()`

  - 칠해지지 않은 칸을 X 로 칠하기 = `remain_off()`

#### Pattern decision algorithm

  1. 

#### Tirivial algorithm

  1. `if (sum(k) + len(k) - 1 == n) line()`

  2. `if (len(k) == 1) if (m == k[0]) remain_off()`