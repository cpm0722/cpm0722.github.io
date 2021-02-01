---

title: "[System Programming] ps 명령어 구현"
subtitle: "ps command"
categories: [System Programming, UNIX]
tags: [System Programming]
permalink: /system-programming/ps-command
date: 2020-09-16 00:00:00 +0000
last_modified_at: 2020-09-16 00:00:00 +0000
project: true

---

## 개요

리눅스 내장 명령어 ps를 구현한 프로그램이다. ps 명령어의 a, u, x 옵션을 사용할 수 있다.

![01.jpg](/assets/images/2020-09-16-PS-Command/01.jpg)

## 실행 방법

```bash
$make pps
./pps
```

## 기능

1. pps

    ![02.jpg](/assets/images/2020-09-16-PS-Command/02.jpg)

2. pps a

    ![03.jpg](/assets/images/2020-09-16-PS-Command/03.jpg)

3. pps u

    ![04.jpg](/assets/images/2020-09-16-PS-Command/04.jpg)

4. pps x

    ![05.jpg](/assets/images/2020-09-16-PS-Command/05.jpg)

5. pps au

    ![06.jpg](/assets/images/2020-09-16-PS-Command/06.jpg)

6. pps ax

    ![07.jpg](/assets/images/2020-09-16-PS-Command/07.jpg)

7. pps ux

    ![08.jpg](/assets/images/2020-09-16-PS-Command/08.jpg)

8. pps aux

    ![09.jpg](/assets/images/2020-09-16-PS-Command/09.jpg)

## 구현 방법

### a u x 옵션

1. a: 다른 사용자들의 프로세스도 보여준다.
    - STAT column 추가
    - CMD column이 COMMAND로 변경
2. u: 터미널이 다른 프로세스도 보여준다.
    - USER, PID, %CPU, %MEM, VSZ, RSS, STAT, START column 추가
    - CMD column이 COMMAND로 변경
3. x: 터미널이 없는 프로세스도 보여준다.
    - STAT column 추가
    - CMD column이 COMMAND로 변경

### process

ps 명령어의 모든 정보들은 Linux File System에서 얻을 수 있다. 아래는 각각의 항목들에 대한 설명과 값의 출처(Linux File System에서의 파일)들을 작성한 것이다. 프로세스를 추상화한 myProc 구조체를 정의하고, 얻어낸 정보들을 통해 각 프로세스 당 하나의 myProc 인스턴스를 생성했다.

```c
//process를 추상화 한 myProc 구조체
typedef struct{
	unsigned long pid;
	unsigned long uid;			//USER 구하기 위한 uid
	char user[UNAME_LEN];		//user명
	long double cpu;			  //cpu 사용률
	long double mem;			  //메모리 사용률
	unsigned long vsz;			//가상 메모리 사용량
	unsigned long rss;			//실제 메모리 사용량
	unsigned long shr;			//공유 메모리 사용량
	int priority;		    		//우선순위
	int nice;					      //nice 값
	char tty[TTY_LEN];			//터미널
	char stat[STAT_LEN];		//상태
	char start[TIME_LEN];		//프로세스 시작 시각
	char time[TIME_LEN];		//총 cpu 사용 시간
	char cmd[CMD_LEN];			//option 없을 경우에만 출력되는 command (short)
	char command[CMD_LEN];	//option 있을 경우에 출력되는 command (long)
}myProc;
```

### column

1. USER: 프로세스 소유자명
    - /proc/pid/stat 파일의 uid 알아낸 뒤 uid에서 유저명 획득

        ```c
        #include <sys/types.h>
        #include <pwd.h>

        struct stat statbuf;
        stat("/proc/pid/stat", &statbuf);
        struct passwd *upasswd = getpwuid(statbuf.st_uid);
        char user[32];
        strcpy(user, upasswd->pwd_name);
        ```

2. PID: 프로세스 ID
    - /proc/pid/stat의 1번째 token
3. CPU: CPU 사용률
    - ((utime+stime) / hertz) / (uptime-(startTime/hertz)) * 100
    - utime: User Mode에서 프로세스가 사용한 jiffies(clock ticks)
        - /proc/pid/stat의 14번째 token
    - stime: Kernel Mode에서 프로세스가 사용한 jiffies(clock ticks)
        - /proc/pid/stat의 15번째 token
    - startTime: OS 부팅 후 프로세스 시작까지의 jiffies(clock ticks)
        - /proc/pid/stat의 22번째 token
    - uptime: OS 부팅 후 지난 시간(second)
        - /proc/uptime의 1번째 token
    - hertz: 1초 간 일어나는 문맥교환 횟수

        ```c
        #include <unistd.h>

        int hertz = (int)sysconf(_SC_CLK_TCK);
        ```

4. MEM: 메모리 사용률
    - RES / memTotal
        - RES: 물리 메모리 사용량
            - /proc/pid/status의 VmHWM(21행)
        - memTotal: 전체 메모리 크기
            - /proc/meminfo의 1행 (kib를 kb로 변환해 사용)
5. VSZ: 가상 메모리 사용량
    - /proc/pid/status의 VmSize(18행)
6. RSS: 실제 메모리 사용량
    - /proc/pid/status의 VmRSS(22행)
7. TTY: 프로세스와 연결된 터미널
    - /proc/pid/fd의 0이 symbolic link로 가리키는 파일명 확인

        ```c
        #include <unistd.h>
        ssize_t readlink(const char *pathname, char *buf, size_t bufsize);
        //return: read byte 수, error일 경우 -1
        ```

    - sudo 권한 없을 경우 fd에 대해 접근 불가능한 몇몇 process 존재
        - /dev 내 문자 디바이스 파일들에 중 statbuf.st_rdev와  /proc/pid/stat의 ttyNr (7번째 token)가 같은 것을 찾아 파일 명 획득
8. STAT: 프로세스 상태(State)
    - /proc/pid/stat 3번째 token
    - 각 문자 별 상세 정보

        ```
        D    uninterruptible sleep (usually IO)
        R    running or runnable (on run queue)
        S    interruptible sleep (waiting for an event to complete)
        T    stopped by job control signal
        t    stopped by debugger during the tracing
        W    paging (not valid since the 2.6.xx kernel)
        X    dead (should never be seen)
        Z    defunct ("zombie") process, terminated but not reaped by its parent

        <    high-priority (not nice to other users)
        N    low-priority (nice to other users)
        L    has pages locked into memory (for real-time and custom IO)
        s    is a session leader
        l    is multi-threaded (using CLONE_THREAD, like NPTL pthreads do)
        +    is in the foreground process group
        ```

        1. None: nice가 0인 경우
            - nice: /proc/pid/stat의 19번째 token
        2. <: nice가 음수인 경우
            - nice: /proc/pid/stat의 19번째 token
        3. N: nice가 양수인 경우
            - nice: /proc/pid/stat의 19번째 token
        4. L: 가상 메모리가 Lock인 경우 (vmLck != 0)
            - vmLck: /proc/pid/status의 19행
        5. s: sid와 pid가 같은 경우 (세션 리더 프로세스인 경우)
            - sid: /proc/pid/stat의 6번째 token
        6. l: 멀티 쓰레드 프로세스인 경우 (num_threads > 1)
            - num_threads: /proc/pid/stat의 20번째 token
        7. +: 전경 프로세스 그룹에 속한 경우 (tgpid != -1)
            - tpgid: /proc/pid/stat의 8번째 token
9. START: 프로세스 시작 시각
    - time(NULL) - (uptime-(startTime/hertz)
        - 24시간 이내
            - 포맷: "%H:%M"
        - 7일 이내
            - 포맷: "%a %d"
        - 그 외
            - 포맷: "%y"
10. TIME: 총 cpu 사용 시각
    - ((utime+stime+cutime+cstime) / hertz)
        - 포맷: [DD-]hh:mm:ss
11. CMD: 프로세스 실행 시 실행된 파일명
    - /proc/pid/stat의 2번째 token
12. COMMAND: 프로세스 실행 시 입력된 명령어
    - proc/pid/cmdline

## [GitHub Link](https://github.com/cpm0722/LSP/tree/master/pps)
