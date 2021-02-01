---

title: "[운영체제] Paging Mechanism"
subtitle: Paging Mechanism
categories: [Operating System]
tags: [Operating System]
permalink: /operating-system/paging-mechanism
date: 2020-10-20 00:00:00 +0000
last_modified_at: 2020-10-20 00:00:00 +0000

---

---

숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다.

# Paging Mechanism

paging 기법에 대해 자세히 알아보자. 위에서 살펴본 고정 분할 및 가변 분할 기법은 각각 내부 단편화, 외부 단편화의 문제점이 존재했다. paging은 이러한 단점들을 해결하기 위해 고안된 방식이다. paging을 사용하면 결론적으로 외부 단편화는 발생하지 않으며, 내부 단편화는 아주 적은 횟수 (대개 process 당 1회) 발생하게 된다.

![01.png](/assets/images/2020-10-20-Paging-Mechanism/01.png)

paging이란 memory 가상화에서 가상 주소와 물리 주소를 mapping시킬 때에 page frame을 단위로 하는 방식이다. page table에는 가상 주소, 물리 주소 뿐만 아니라 P, M, U bit 등의 control bit도 담겨져 있다. P(present) bit는 해당 page가 memory에 위치하는가에 대한 bit로, 1일 경우 가리키는 물리 주소가 물리 memory 영역이라는 의미이고 0일 경우에는 가리키는 물리 주소가 memory가 아닌 disk의 swap 영역이라는 뜻이다. 즉, P bit가 0일 경우에는 swap 영역에 있는 page를 memory로 불러와야 한다. 이러한 과정을 **page fault**라고 한다. page fault는 결국 disk I/O를 호출하는 것이기에 schedule() 함수를 호출한다. 한편 M(modify) bit는 해당 page가 수정된 적이 있는지에 대한 bit이고, W(write) bit, D(dirty) bit라고도 불린다. U(used) bit는 해당 page를 read한 적이 있는지에 대한 bit로, R(read) bit라고도 불린다. page table은 OS가 각각의 process에게 개별적으로 부여하게 되며,  task_struct와 같은 PCB들은 멤버 변수로 page table을 가리키는 포인터 값을 저장한다. 한편, 대부분의 가상 memory 기법은 page table을 실제 memory가 아닌 가상 memory에 저장하게 된다. process가 running 상태라면, 최소한 해당 process의 page table 중 일부분은 memory에 존재해야 하고, 전체 page table이 memory에 존재하는 것이 가장 바람직할 것이다.

# Virtual Address

![02.png](/assets/images/2020-10-20-Paging-Mechanism/02.png)

page table을 통해 사용되는 가상 주소와 물리 주소는 모두 number + offset의 구조를 갖는다. page number를 통해 page table의 몇 번째 row에 접근할 지를 파악하고, register에 저장된 page table의 포인터 값과 page number를 더해 해당 page table의 row에 접근한다. 이후 얻은 frame number를 통해 실제 물리 memory에 접근하게 된다. 하지만 frame number는 결국 물리 memory에서의 시작 주소를 뜻하는 값이기 때문에 얼마나 data를 읽어들일지에 대한 정보는 알지 못한다. 이 때 사용하는 것이 offset이다. 가상 주소에서의 offset을 그대로 물리 주소에서 사용하게 된다. 이러한 모든 작업은 대개 HW(CPU의 Memory Management Unit)가 수행하게 된다. 과거에는 OS에서 SW를 통해 구현해 사용하기도 했으나 속도가 HW를 이용하는 것에 비해 많이 느리다.

가상 주소의 bit 사용량을 통해 역으로 OS의 각종 변수 값을 유추할 수도 있다. 가상 주소에서 offset이 차지하는 bit수가 $o$라면, 해당 OS의 page frame size는 $2^o$가 된다. 한편, 가상 주소에서 page number가 사용하는 bit 수가 $p$라면, 해당 OS의 page table의 최대 크기(가질 수 있는 최대 항목 수)는 $2^p$가 된다.

![03.png](/assets/images/2020-10-20-Paging-Mechanism/03.png)

# 과도한 크기의 Page Table 문제 해결

## 계층 구조 Page Table 사용

page table의 크기는 page table entry의 size * page table가 가질 수 있는 최대 항목 수로 계산할 수 있다. 즉, page table이 가질 수 있는 최대 항목 수가 클 수록 page table의 크기도 커진다는 것이다. 너무 큰 page table을 운용하게 되면 memory 낭비가 심해진다. 각 process마다 page table 운용을 위해 여러 page frame을 사용하지만 그 중 실제로 page table의 극히 일부만 사용하는 상황이 대표적인 예시이다. 이를 해결하기 위한 대표적인 방법이 계층 구조 page table이다. 주로 2단계 계층 구조, 3단계 계층 구조 등이 있다. 우선 page directory가 있어 각각의 항목이 page table을 가리키도록 한다. page directory가 가리키는 page table이 꽉 찼을 경우에만 page directory의 다음 항목에서 새로운 page table을 가리키도록 동적으로 운용하는 방식이다. 아래는 2단계 계층 구조 page table의 예시이다.

![04.png](/assets/images/2020-10-20-Paging-Mechanism/04.png)

## Inverted Page Table 사용

Page Number를 그대로 사용하지 않고 hash function을 이용해 얻은 hash value로 사용하게 된다. hash value는 hash table에서의 인덱스이다. hash table의 항목 수는 물리 memory의 page frame의 개수와 동일하다. 즉, hash table은 모든 process가 공용으로 사용하는 것이다. 따라서 hash table entry에는 page number뿐만 아니라 pid까지 함께 담겨져 있다. hash table에서의 collision을 해결하기 위해 linked list로 다음 entry를 연결하게 된다. 이렇게 찾은 hash table entry의 hash table에서의 인덱스를 이용해 page frame을 찾아가게 된다. hash table에서의 인덱스가 $i$라면, mapping된 page frame도 실제 물리 memory에서 $i$번째 page frame이 된다.

![05.png](/assets/images/2020-10-20-Paging-Mechanism/05.png)

## TLB (Translation Look-aside Buffer) 사용

translation look-aside buffer란 page table 항목들을 저장하기 위한 특수한 고속 cache 장치이다. 가장 최근에 참조된 n개의 page table entry항목들을 저장하게 된다. page table은 기존과 동일하게 control bits와 frame number를 저장하고 있으며, page number를 page table에서의 인덱스로 사용한다. 하지만 TLB에서는 page number를 인덱스로 사용해 각 항목에 접근할 수 없기 때문에 page table의 항목들이 가진 정보에 더해 page number를 추가적으로 저장해야 한다. 이를 **연관 사상(Assosiative Mapping)**이라고 한다.

실제 가상 주소를 물리 주소로 변환하는 과정을 따라가보자. 가상 주소가 주어지면 우선 TLB에서 해당 page number가 있는지 확인한다. page number가 TLB에 있을 경우 TLB Hit으로, 바로 frame number를 얻어 물리 주소를 구해낸다. 만약 TLB에 page number가 없을 경우 TLB Miss로, 기존과 동일하게 page table에서 page number를 통해 frame number를 구해낸다. 이후 해당 page number에 관련된 정보들을 TLB에 추가한다.  만약 TLB에 여유 공간이 없을 경우 가장 오래된 항목을 제거해 공간을 확보한다. 한편, 만약 page table에서 P bit가 0이라면 Page Fault로, secondary memory(swap)에 접근해 해당 page frame을 memory로 load한다. 이후 다시 page table에서 물리 주소를 찾아나선다.

![06.png](/assets/images/2020-10-20-Paging-Mechanism/06.png)

# 적절한 Page (Frame) Size

page의 크기는 HW 설계에 있어서 매우 중요한 issue 중 하나이다. 여러 관점에서 장단점을 고려해서 신중하게 결정해야만 한다. 만약 page의 크기가 작다면 내부 단편화가 적게 발생할 것이다. 대신 한 process 당 필요한 page의 수가 많아지고, 이는 결국 page table의 size를 늘리는 결과를 낳는다. page table의 size가 커지면 Multi-Programming 환경에서는 여러 활성 process 중 일부의 page table이 물리 memory가 아닌 swap 영역에 있어야 함을 의미한다. 최악의 경우에는 한 번의 memory 참조로 Page Fault가 2번(page table, page frame) 발생할 수 있는 것이다. 

# Paging Replacement Policy

Memory의 모든 page frame이 사용 중인 상황에서 swap에 위치한 page frame을 참조하는 상황이 발생할 수 있다. 이 때에는 memory의 page frame 중 하나를 swap의 page frame과 교체해야 한다. 이러한 현상을 Page Fault라고 부른다. page fault가 다수 발생하는 현상을 thrashing이라고 한다. thrashing을 방지하기 위해 합리적인 page replacement policy를 채택해야 한다. 만약 자주 호출되는 page frame을 memory에서 빼내어 swap으로 이동시키게 되면, page fault 발생 횟수가 증가해 성능에 악영향을 미칠 것이다. page replacement 정책의 핵심은 page frame이 미래에 얼마나 참조될 지를 예측하는 것이다. 미래의 일을 완전히 예견하는 것은 불가능하나 과거의 경향을 근거로 미흡하게나마 예측할 수는 있다. 따라서 page replacement 정책은 대개 과거의 page frame 이동의 경향을 파악해 미래를 예측하고자 한다. 하지만 너무 정교한 page replacement policy를 적용하게 된다면 오히려 HW와 SW 상의 부담이 더 커지기 때문에 적절한 trade-off가 이루어져야 한다. 아래에서는 6가지 Paging Replacement 정책에 대해 살펴본다.

## Optimal

미래에 참조될 page의 순서를 모두 아는 상태에서 앞으로 참조될 때까지의 시간이 가장 긴 page를 교체한다. 당연하게도 현실에서는 구현할 수 없다. optimal 정책은 어디까지나 다른 paging replacement 정책을 평가하는 기준으로써의 가치만 있을 뿐, 구현 대상이 아니다.

![07.png](/assets/images/2020-10-20-Paging-Mechanism/07.png)

5 time에서 page 5가 삽입되는데, 이전 time 기준으로 main memory에 존재하는 page 2, 3, 1는 각각 1 time, 4 time, $\infin$ time 후에 다시 참조된다. 따라서 이후 참조되기까지의 시간이 가장 많이 남은 page 1이 교체되게 된다.

## FIFO (First Input First Out)

먼저 들어온 page가 먼저 나가는 단순 Queue 방식이다. scheduling 중 RR과 비슷하다고 볼 수 있다. 가장 오래 전에 반입된 page는 memory에 가장 오래 존재했기 때문에 더이상 사용되지 않을 것이라는 논리 하에서 구현된 정책이다. 구현이 매우 간단하지만 좋은 성능을 보이지 못한다. FIFO 정책 하에서 main memory의 page frame 수를 늘릴 경우에는 page fault가 덜 발생할 것 같지만, 의외로 page fault가 더 자주 발생하기도 한다. 이를 FIFO abnormally(이상 현상)이라고 한다.

![08.png](/assets/images/2020-10-20-Paging-Mechanism/08.png)

## LIFO (Last Input Last Out)

가장 최근에 들어온 page가 빠져나가는 Stack 방식이다. 하나의 page frame만이 지속적으로 교체되기 때문에 page fault가 매우 자주 발생할 것이다. 하지만 의외로 평균적인 성능은 FIFO와 비슷하다.

![09.png](/assets/images/2020-10-20-Paging-Mechanism/09.png)

## LRU (Least Recently Used)

가장 오랫동안 참조되지 않은 page를 교체하는 것이다. LRU는 Optimal과 가장 비슷한 성능을 보이지만 실제로는 구현이 매우 곤란하다는 단점이 있다. 각 page frame마다 가장 최근에 참조된 시각을 기록해야 하는데, 결국 물리 memory 내의 모든 page frame에 대해 int 변수를 추가하고 매 참조마다 갱신하는 형태가 될 수 밖에 없다. 이는 시스템에 큰 부하를 줘 좋은 성능을 내지 못한다.

![10.png](/assets/images/2020-10-20-Paging-Mechanism/10.png)

5 time에서 page 5가 삽입되는데, 이전 time 기준으로 memory에 존재하는 page 2, 3, 1은 각각 참조 시점이 2 time, 3 time, 1 time 전이다. 따라서 가장 오래 전에 참조된 page 3이 교체된다.

## LFU (Least Frequently Used)

참조된 빈도가 가장 낮은 page를 교체하는 것이다. LRU와 동일하게 좋은 성능을 보이지만 구현하기 곤란하다. LRU와 마찬가지로 각 page frame마다 새로운 변수를 추가해야 하는데, 이 경우에는 참조 횟수일 것이다. 만약 동일한 참조 횟수를 가진다면 FIFO 정책을 채택해 먼저 들어온 page를 교체한다.

![11.png](/assets/images/2020-10-20-Paging-Mechanism/11.png)

5 time에서 page 5가 삽입되는데, 이전 time 기준으로 memory에 존재하는 page 2, 3, 1은 각각 참조 횟수 2, 1, 1을 갖는다. 따라서 page 3과 1 중 선택을 해야 하는데, FIFO 정책을 채택해 더 먼저 들어온 page 3을 교체한다.

## Clock = NUR (Not Used Recently)

Clock 정책은 현대 OS에서 채택하고 있는 page replacement 정책이다. LRU나 LFU와 같이 추가적인 변수를 생성하지 않고, 기존에 page table에 이미 존재하던 R, W bit를 활용하게 된다. 사용하는 bit 수가 더 많아질수록 더 좋은 성능을 보인다. 아래에서는 2 bit를 사용하는 two handed clock이 아닌 one handed clock의 예시이다. 교체하는 우선 순위는 다음과 같다.

1. 참조되지 않았으며, 수정되지 않음 (R = 0, W = 0)
2. 참조되었으며, 수정되지 않음 (R = 1, W = 0)
3. 참조되었으며, 수정됨 (R = 1, W = 1)

우선 순위 1을 먼저 찾고, 우선 순위 1이 없을 경우 우선 순위 2를 찾아나가되 그 과정에서 지나치는 모든 page frame의 R bit를 0으로 설정한다. 만약 우선 순위 2도 없을 경우 모든 page frame을 탐색하면서 R bit를 0으로 만들었을 것이다. 그 상태에서 다시 우선 순위 1을 찾는 반복을 수행한다.

![12.png](/assets/images/2020-10-20-Paging-Mechanism/12.png)

새로운 symbol이 추가되는데, $\rightarrow$는 memory를 가리키는 pointer이다. clock 정책에서 다음에 삽입할 page가 어디인지를 가리킨다. * symbol은 참조 여부이다. *가 있을 경우 참조된 frame (R bit = 1), *가 없을 경우 참조되지 않은 frame (R bit = 0)이다.

5 time에서 page 5가 삽입되는데, 현재 pointer가 가리키는 page frame은 2이다. page 2는 이미 참조가 된 상태이므로 우선 순위 1에 해당되지 않는다. 따라서 우선 순위 2를 찾아 나선다. 그 과정에서 page 2, 3, 1의 R bit를 0으로 수정한다. 우선 순위 2를 찾기 실패했기 때문에 처음으로 되돌아온다. page 2가 R bit=0이 되었기 때문에 우선 순위 1에 해당한다. 따라서 page 2를 교체한다.

6 time에서 page 2가 삽입되는데, 현재 pointer가 가리키는 page frame은 3이다. page 3은 R bit=0이므로 우선 순위 1에 해당되기 때문에 3 page를 교체한다.

7 time에서 page 4가 삽입되는데, 현재 pointer가 가리키는 page frame은 1이다. page 1은 R bit=0이므로 우선 순위 1에 해당되기 때문에 1 page를 교체한다.

8 time에서 page 5가 삽입되는데, 이미 main memory에 존재하므로 fault가 발생하지 않는다. 따라서 pointer도 이동하지 않는다.

9 time에서 page 3이 삽입되는데, 현재 pointer가 가리키는 page frame은 5이다. page 5는 이미 참조가 된 상태이므로 우선 순위 1에 해당되지 않는다. 따라서 우선 순위 2를 찾아 나선다. 그 과정에서 page 5, page 2, page 4의 R bit를 0으로 수정한다. 우선 순위 2를 찾기 실패했기 때문에 처음으로 되돌아온다. page 5가 R bit=0이 되었기 때문에 우선 순위 1에 해당한다. 따라서 page 5를 교체한다.

10 time에서 page 2가 삽입되는데, 이미 main memory에 존재하므로  fault가 발생하지 않는다. 그런데 R bit=0이므로 R bit=1로 변경한다.

11 time에서 page 5가 삽입되는데, 현재 pointer가 가리키는 page frame은 2이다. page 2는 이미 참조가 된 상태이므로 우선 순위 1에 해당되지 않는다. 따라서 우선 순위 2를 찾아 나선다. 우선 순위 1에 해당하는 page 4를 찾았고, 그 과정에서 page 2의 R bit를 0으로 변경했다. page 4를 교체한다.

12 time에서 page 2가 삽입되는데, 이미 main memory에 존재하므로 fault가 발생하지 않는다. 그런데 page 2의 R bit=0이므로 R bit를 1로 변경한다.
