---

title: "[운영체제] Introduction to Operating Systems"
subtitle: Introduction to Operating Systems
categories: [Operating System]
tags: [Operating System]
permalink: /operating-system/introduction-to-operating-systems
date: 2020-09-07 00:00:00 +0000
last_modified_at: 2020-09-07 00:00:00 +0000

---

---

숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다.

# OS (Operating System)

## OS의 정의

OS는 SW의 일종이다.  '자원'(HW)관리'자'(SW)로 정의할 수 있다. HW를 SW로 관리해주는 역할을 수행한다. 구체적으로 CPU, Memory, Disk 등을 struct로 정의해 각각 Process, Virtual Memory, File System을 만들어낸다.  OS는 kernel 함수를 이용해 HW를 관리하며, 사용자는 kernel 함수를 직접 호출하지 않고, system call을 사용하게 된다.

## OS의 역할

1. 사용자가 프로그램을 쉽게 사용할 수 있도록 돕는다.

    실제로 HW가 어떻게 작동하는지를 숨기고(가상화), 사용자가 간단하게 사용할 수 있는 관리 도구를 제공(추상화)한다.

2. 시스템이 정확하고 효율적으로 작동하는지 확인한다.

## 추상화 vs 가상화

- 추상화: HW를 SW로 표현하는 것
    - CPU → Process
    - Memory → Virtual Memory
- 가상화: 실제 생김새나 기능을 사용자에게 숨겨 구조 등을 착각하게 만듦
    - CPU → 프로그램마다 별도의 CPU를 모두 보유한 것처럼 보이게 함 (Context Switch)
    - Memory → 프로그램마다 별도의 Memory를 모두 보유한 것처럼 보이게 함 (Share Memory)

# CPU 가상화

Context Switch(문맥 교환)이란 아주 빠른 속도(대개 10ms)로 하나의 CPU가 process들을 interleave하게 실행하는 것이다. 사용자는 각 process들이 동시에 실행되는 것으로 착각하게 된다. 이를 통해 Multi-tasking이 구현되는 것이다.

문맥 교환이 발생하는 시점은 3가지로 구분된다.

1. TQ (문맥 교환 발생하는 시간이 충족됐을 때)
2. exit()
3. DISK I/O

위의 경우가 오게 되면 `switchto()` 함수를 호출해 문맥 교환을 수행하게 된다. 인자로는 두 `task_struct`를 받는다.

```c
switchto([task_struct of running process], [task_struct of next process])
```

다음 수행할 process의 `task_struct`는 `schedule()` 함수의 return으로 얻을 수 있다.

문맥 교환은 여러 process들을 번갈아가며 수행하는데, 이 때 전환되는 process의 순서는 일정하지 않을 수 있다. 예를 들어 process가 A → B → C → D 순으로 문맥 교환이 발생했다고 하더라도, 다음에도 같은 순서로 process가 실행된다고 보장할 수 없다.

# Memory 가상화

같은 프로그램 파일(disk에 위치)로 두 process(memory에 위치)를 실행하게 되면 두 process가 동일한 memory 주소를 참조하는 것처럼 보이는 상황이 있을 수 있다. 이는 실제 물리적으로 같은 memory를 참조하는 것이 아니라 각각의 process가 갖고 있는 가상 memory 주소 상에서 같은 주소를 참조하는 것이다. 각 process의 가상 memory 주소는 시작 주소가 다를 것이므로, 가상 memory 주소가 같다고 하더라도 실제 물리 memory 주소는 같지 않다. OS는 이렇게 가상의 memory 주소와 물리적 memory 주소를 mapping하는 역할을 수행한다. (최근의 CPU에는 이 역할을 OS 대신 수행해주는 MMU 기능이 들어가 있다. SW인 OS보다 HW인 CPU가 처리하는 것이 더 빠르기 때문이다.) 가상 Memory 주소 공간은 register 크기에 따라 달라진다(32bit, 64bit). 한편, 가상 memory는 가상 Memory 주소 공간을 의미하는 것이 아닌, 실제 Memory + SWAP(Disk에 위치하지만 Memory로 임시로 사용되는 영역, Linked LIst로 구현됨)을 뜻한다.

# Concurrency (동시 실행)

process 간 전환이 일어날 때, 항상 원자적(atomic)으로 실행되지는 않는다. 따라서 같은 조건에서 같은 횟수로 process를 실행했다고 하더라도 따라서 카운터의 값이 항상 같지는 않다. load(memory→register)와 ++ 연산, save(register→memory)의 과정이 원자적으로 일어나지 않기 때문이다.
