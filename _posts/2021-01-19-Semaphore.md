# Semaphore
title: Semaphore
subtitle: Semaphore
categories: "Operating System"
tags: "Operating System"
date: 2021-01-19 19:11:49 +0000
last_modified_at: 2021-01-19 19:11:49 +0000
---

Created: Nov 17, 2020 5:06 PM
Reference: Jiman Hong: Soongsil Univ., Operating Systems Concepts 10th by A.Silberschatz P.Galvin and G.Gagne
status: completed

```yaml
cleanUrl: /os/semaphore
disqus: true
```

숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다.

# Semaphore

semaphore는 다수의 thread 사이의 병행성 유지를 위해 OS 단위에서 제공되는 기법이다. 기본적인 작동 원리는 특정 thread가 특정 signal을 수신할 때까지 정해진 위치에서 wait하도록 강제하는 것이다. 

## counting semaphore

counting semahpore는 정수값을 갖는 counting 변수와 3가지 연산으로 구성된다. 범용 semaphore라고도 불리운다. 3가지 연산은 아래와 같다.

1. semInit(): semaphore 변수를 음이 아닌 값(대개 1)으로 초기화를 수행한다.
2. semWait(): semaphore 변수 값을 감소시킨다. 만약 값이 음수가 되면 semWait()을 호출한 thread는 block된다. 그 외에는 해당 thread는 정상적으로 계속 수행한다.
3. semSignal(): semaphore 변수 값을 증가시킨다. 만약 값이 양수가 아니면 semWait()에 의해 block된 thread 중 하나를 깨운다.

기본적인 pseudo code는 다음과 같다.

```c
typedef struct{
	int count;
	queue waitQueue;
} semaphore;

void semInit(semaphore s, int n)
{
	s.count = n;
}

void semWait(semaphore s)
{
	s.count--;
	if(s.count < 0){
		//요청한 thread를 s.waitQueue에 push
		//요청한 thread의 상태를 block으로 변경
	}
}

void semSignal(semaphore s)
{
	s.count++;
	if(s.count <= 0){
		//s.waitQueue에서 thread 1개를 pop
		//pop한 thread의 상태를 runnable로 변경 후 OS의 readyQueue에 push
	}
}
```

count 변수는 값이 음수인 경우에 그 절대값은 대기 queue의 길이를 의미한다.

## binary semaphore (mutex)

mutex는 semaphore 변수가 0 또는 1의 binary 값만 갖는 semaphore를 뜻한다. 동일하게 3가지 연산으로 구성된다.

1. semInitB(): semaphore 변수를 0 또는 1로 초기화한다.
2. semWaitB(): semaphore 변수 값을 확인해 0일 경우 semWaitB()를 호출한 thread는 block되고, 1일 경우 값을 0으로 변경시킨 뒤 thread는 계속 수행한다.
3. semSignalB(): block된 thread가 있는지 확인한 후, 만약 있을 경우 해당 thread들 중 하나를 깨우고, 없을 경우 semaphore 변수 값을 1로 설정한다.

pseudo code는 다음과 같다.

```c
typedef struct{
	_Bool value;
	queue waitQueue;
}binary_semaphore;

void semInitB(binary_semaphore s, int n)
{
	s.value = n;
}

void semWaitB(binary_semaphore s)
{
	if(s.value == 1)
		s.value = 0;
	else{
		//요청한 thread를 s.waitQueue에 push
		//요청한 thread의 상태를 block으로 변경
	}
}

void semSignalB(binary_semaphore s)
{
	if(s.waitQueue.empty())
		s.value = 1;
	else{
		//s.waitQueue에서 thread 1개를 pop
		//pop한 thread의 상태를 runnable로 변경 후 OS의 readyQueue에 push
	}
}
```

binary semahpore는 일반 범용 semaphore에 비해 구현이 간단하다는 장점이 있다. 둘 모두 waitQueue를 운용한다는 점에서 공통적이다.

## strong / weak semaphore

queue에서 FIFO 방식을 사용하는 semaphore를 강성(strong) semaphore라고 하고, 특별히 queue의 순서를 명시하지 않은 semaphore를 약성(weak) semaphore라고 한다. 하지만 실제로 대부분의 OS에서는 강성 semaphore를 사용한다. starvation이 없고, 직관적이며 구현하기도 용이하기 때문이다.

아래는 강성 semaphore의 예시이다. D thread는 생산자,와 A, B, C thread는 소비자인 문제이다. 초기 semaphore 변수 s가 값이 1로 시작된다.  s의 값이 음수일 때에는 그 절댓값이 기다리는 thread의 개수(waitQueue 내 thread의 개수)를 뜻하고, s의 값이 음수가 아닐 때에는 생산자가 생성한 자원의 여분 개수를 뜻한다.

![Semaphore%20b9f4eb0a53d84d5d8095e6194255911c/Untitled.png](Semaphore%20b9f4eb0a53d84d5d8095e6194255911c/Untitled.png)

## mutual exclusion problem

범용 semaphore를 사용해 상호 배제 문제를 해결해보자. 상호 배제 문제란 동일한 자원에 접근하려는 n개의 thread의 병행성을 처리하는 문제이다. semInit()에서 count 변수를 0이 아닌 변수로 초기화한다. count 변수의 초기값은 자원의 개수를 의미한다. 따라서 count 변수는 0으로 초기화 되어서는 안된다. 모든 thread가 무한히 block될 것이기 때문이다. 각 thread에서 critical section(임계 영역)을 생성하게 되는데, critical section이란 한 번에 최대 1개의 thread만이 접근할 수 있는 영역이다. semWait()~semSignal() 사이의 영역이 된다. pseudo code는 다음과 같다.

```c
semaphore s;

void thread_execute(int thread_no)
{
	while(1){
		semWait(s);
		//임계 영역
		semSignal(s);
		//임계 영역 이후
	}
}

int main(void)
{
	semInit(s, 1);
	for(int i = 0; i < num_of_threads; i++)
		thread_start(i);
}
```

thread가 1, 2, 3 순서대로 실행된다고 가정했을 때 각 thread는 아래와 같은 형태로 실행된다.

![Semaphore%20b9f4eb0a53d84d5d8095e6194255911c/Untitled%201.png](Semaphore%20b9f4eb0a53d84d5d8095e6194255911c/Untitled%201.png)

## producer-consumer problem

생산자-소비자 문제를 mutex를 이용해 해결해보자. 생산자-소비자 문제는 다수의 생산자 thread가 각자 자원을 생성해 공용 buffer에 저장하고, 다수의 소비자 thread가 공용 buffer에서 자원을 1개씩 소비하는 상황의 병행성을 처리하는 문제이다. 공용 buffer에는 한 번에 1개의 thread만 접근 가능하다(critical section)는 조건이 있다. 우선 공용 buffer가 무한한 크기를 갖는다고 가정한다. 이 때 in과 out이라는 pointer 변수를 사용하는데, in은 다음에 생산자가 생성한 자원이 저장될 buffer에서의 위치이며, out은 다음에 소비자가 소비할 자원이 저장된 buffer에서의 위치이다. 따라서 out<in인 경우에만 소비자가 소비할 자원이 있는 것이다. 전체 pseudo code는 다음과 같다.

```c
int n;                     //in-out의 값
binary_semaphore s;        //buffer의 접근을 제어하는 mutex
binary_semaphore delay;    //buffer가 비었는지를 확인해 소비를 제어하는 mutex

void producer(void)
{
	while(1){
		val = produce();       //자원 생산
		semWaitB(s);
		/*
			                     critical section start
		*/
		append(val);           //buffer에 push
		n++;
		if (n == 1)            //buffer.empty()==false가 된 상황
			semSignalB(delay);   //consumer 중 1개 block 해제
		/*
		                       critical section end
		*/
		semSignalB(s);
		}
	}
}

void consumer(void)
{
	//consumer가 producer보다 먼저 실행되는 상황(buffer.empty()==true)를 막기 위해 block
	semWaitB(delay);
	while(1){
		semWaitB(s);
		/*
		                     critical section start
		*/
		val = take();        //buffer에서 pop
		n--;
		/*
			                	critical section end
		*/
		semSignalB(s);
		consume(val);       //자원 소비
		if(n == 0)          //buffer.empty()==true가 된 상황
			semWaitB(delay); //thread block
	}
}

int main(void(
{
	n = 0;
	semInitB(s, 1);
	semInitB(delay, 0);
	for(int i = 0; i < num_of_producers; i++)
		thread_start(i);
	for(int i = 0; i < num_of_consumers; i++)
		thread_start(i);
}
```

위의 code는 producer와 consumer 내의 while문이 매 번 atomic하게 전체가 함께 실행되면 정상적으로 작동할 것이다. 하지만 while loop가 1번 도는 사이에 scheduling이 발생하지 않을 것이라는 보장이 없다. 만약 consumer에서 semSignalB(s)와 if(n==0) 사이에서 scheduling이 발생해 producer가 실행된다면 n은 0에서 1로 변경될 것이고, 그렇다면 다시 scheduling이 되어 consumer로 돌아왔을 때 if(n==0)을 만족하지 못하므로 semWaitB(delay)가 실행되지 않을 것이다. 이는 소비자가 한 개의 thread라면 큰 문제가 되지 않지만, 다수의 thread일 경우에는 문제 상황이 된다. empty임에도 여러 소비자 thread 모두 block되지 않을 수 있기 때문이다.

### solution 1: 보조 변수 사용

```c
int n;                     //in-out의 값
binary_semaphore s;        //buffer의 접근을 제어하는 mutex
binary_semaphore delay;    //buffer가 비었는지를 확인해 소비를 제어하는 mutex

void producer(void)
{
	while(1){
		val = produce();       //자원 생산
		semWaitB(s);
		/*
			                     critical section start
		*/
		append(val);           //buffer에 push
		n++;
		if (n == 1)            //buffer.empty()==false가 된 상황
			semSignalB(delay);   //consumer 중 1개 block 해제
		/*
		                       critical section end
		*/
		semSignalB(s);
		}
	}
}

void consumer(void)
{
	**int m;**
	//consumer가 producer보다 먼저 실행되는 상황(buffer.empty()==true)를 막기 위해 block
	semWaitB(delay);
	while(1){
		semWaitB(s);
		/*
		                     critical section start
		*/
		val = take();        //buffer에서 pop
		n--;
		**m = n;**
		/*
			                	critical section end
		*/
		semSignalB(s);
		consume(val);       //자원 소비
		**if(m == 0)**          //buffer.empty()==true가 된 상황
			semWaitB(delay); //thread block
	}
}

int main(void(
{
	n = 0;
	semInitB(s, 1);
	semInitB(delay, 0);
	for(int i = 0; i < num_of_producers; i++)
		thread_start(i);
	for(int i = 0; i < num_of_consumers; i++)
		thread_start(i);
}
```

n의 값이 변경되는 것을 막기 위해 critical section 내에서 보조 변수 m에 현재 n의 값을 임시로 저장한다. 이후 critical section 밖의 if문에서 n 대신 m이 0인지를 확인하게 된다.

### solution 2: 범용 semaphore 사용

binary semaphore가 아닌 범용 semaphore를 사용하면 애초에 위의 문제 상황이 발생하지 않는다.

```c
semaphore s;               //buffer의 접근을 제어하는 semaphore
semaphore n;               //buffer에 들어있는 자원의 개수를 제어하는 semaphore

void producer(void)
{
	while(1){
		val = produce();       //자원 생산
		semWait(s);
		/*
			                     critical section start
		*/
		append(val);           //buffer에 push
		semSignal(n);          //consumer 중 1개 block 해제
		/*
		                       critical section end
		*/
		semSignal(s);
		}
	}
}

void consumer(void)
{
	while(1){
		semWait(n);          //buffer.empty()==true일 때 실행되는 것을 방지하기 위해 block
		semWait(s);
		/*
		                     critical section start
		*/
		val = take();        //buffer에서 pop
		/*
			                	critical section end
		*/
		semSignal(s);
		consume(val);       //자원 소비
	}
}

int main(void(
{
	semInit(s, 1);
	semInit(n, 0);
	for(int i = 0; i < num_of_producers; i++)
		thread_start(i);
	for(int i = 0; i < num_of_consumers; i++)
		thread_start(i);
}
```

공용 변수 n과 delay를 통합해 하나의 범용 semaphore n으로 운용한다. n의 값에 따라 delay를 wait 또는 signal하지 않고 무조건적으로 producer에서는 semSignal(n), consumer에서는 semWait(n)하게 된다. n은 buffer에 들어가 있는 자원의 개수(음수일 경우 그 절댓값은 waitQueue에 들어있는 thread의 수)임과 동시에 thread의 실행 순서를 제어하는 역할을 하게 된다. consumer thread들은 매 번 실행될 때마다 semWait(n)을 하게 된다. 따라서 producer에서 semSignal(n)과 semSignal(s)가 서로 순서가 바뀌어 semSignal(n)이 critical section 밖에서 수행된다고 하더라도 동일하게 실행된다. 왜냐하면 어차피 consumer들은 semWait(n)을 통해 block된 상태이기에 semSignal(n)이 호출되어야 수행될 수 있기 때문이다.

### 유한 buffer 사용

위의 모든 solution은 무한한 buffer를 사용한다는 가정 하에서 이루어졌다. 하지만 실제로 무한한 buffer는 존재하지 않으므로 유한한 buffer를 사용하게 된다. 대개 circular queue를 사용하게 된다. pseudo code는 다음과 같다.

```c

semaphore s;               //buffer의 접근을 제어하는 semaphore
semaphore n;               //buffer에 들어있는 자원의 개수를 제어하는 semaphore
semaphore b;               //유한 buffer를 관리하는 semaphore

void producer(void)
{
	int val;
	while(1){
		val = produce();       //자원 생산
		**semWait(b)**;            //buffer.full()==true일 경우 block
		semWait(s);
		/*
			                     critical section start
		*/
		append(val);           //buffer에 push
		semSignal(n);          //consumer 중 1개 block 해제
		/*
		                       critical section end
		*/
		semSignal(s);
		}
	}
}

void consumer(void)
{
	int val;
	while(1){
		semWait(n);          //buffer.empty()==true일 때 실행되는 것을 방지하기 위해 block
		semWait(s);
		/*
		                     critical section start
		*/
		val = take();        //buffer에서 pop
		/*
			                	critical section end
		*/
		semSignal(s);
		**semSignal(b)**;       //buffer.full()==false이므로 block된 producer 중 1개 unblock
		consume(val);       //자원 소비
	}
}

int main(void(
{
	semInit(s, 1);
	semInit(n, 0);
	semInit(b, BUFFER_SIZE);
	for(int i = 0; i < num_of_producers; i++)
		thread_start(i);
	for(int i = 0; i < num_of_consumers; i++)
		thread_start(i);
}
```