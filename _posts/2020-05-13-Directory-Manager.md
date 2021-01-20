---

title: "[System programming] 디렉터리 관리 프로그램 구현"
subtitle: "Directory Manager"
categories: [System Programming, UNIX]
tags: [System Programming]
date: 2020-05-13 00:00:00 +0000
last_modified_at: 2020-05-13 00:00:00 +0000

---

[GitHub Link](https://github.com/cpm0722/LSP/tree/master/dir_manager)

## 개요

디렉터리 관리 프로그램이다. 지정한 특정 디렉터리를 지속적으로 모니터링하고, 해당 디렉터리의 하위 디렉터리 및 파일들의 변경 사항을 로그 파일에 기록한다. 삭제(trash 디렉터리로 이동), 복구(trash 디렉터리에서 복원), 디렉터리 크기 출력, 디렉터리 전체 구조 출력 등의 기능으로 이루어져 있다.

## 실행 방법

```bash
$ make dir_manager
./dir_manager [TARGET_DIRECTORY_PATH]
```

- TARGET_DIRECTORY_PATH: 모니터링을 수행할 타겟 디렉터리의 절대/상대 경로

## 기능

1. 로그 기록

    데몬 프로세스를 사용해 프로그램 실행 시 인자로 입력받은 디렉터리에 대해 지속적으로 모니터링을 수행한다. 모니터링 내역은 현재 작업 디렉터리의 log.txt 파일에 기록한다.

    - 실행 예시

        ![01.jpg](/assets/images/2020-05-13-Directory-Manager/01.jpg)

        ![02.jpg](/assets/images/2020-05-13-Directory-Manager/02.jpg)

2. 휴지통

    '휴지통' 역할의 trash 디렉터리를 현제 작업 디렉터리에 생성한다. trash 디렉터리 하위에 files, info의 2개의 서브 디렉터리가 생성되고, files 디렉터리는 dir_manager의 DELETE 명령어로 삭제한 파일을 저장한다. info 디렉터리는 dir_manager의 DELETE 명령어로 삭제한 파일의 상세 정보(절대 경로, 삭제 시각, 최종 수정 시각)을 저장한다. info 디렉터리의 크기가 2KB를 초과할 경우 가장 오래된 파일부터 삭제를 진행하게 되는데, 이 때 trash/files 디렉터리의 파일도 함께 삭제된다.

    - 실행 예시 (trash/info가 2KB를 초과해 1.txt가 trash에서 삭제됨)

        ![03.jpg](/assets/images/2020-05-13-Directory-Manager/03.jpg)

        ![04.jpg](/assets/images/2020-05-13-Directory-Manager/04.jpg)

3. DELETE

    타겟 디렉터리 내의 파일을 지정한 시각에 삭제하는 명령어이다.

    - 명령어

        ```bash
        $ DELETE [FILENAME] [END_TIME] [OPTION]
        ```

        - FILENAME: 삭제할 파일의 절대/상대 경로
        - END_TIME: 파일을 삭제할 예정 시각 (형식: YYYY-MM-DD HH:MM), 생략 시 바로 삭제
        - OPTION
            - -i : trash 디렉터리를 거치지 않고 바로 삭제
            - -r : 지정한 시각에 삭제 여부 확인 메세지 출력
    - 실행 예시

        ![05.jpg](/assets/images/2020-05-13-Directory-Manager/05.jpg)

        ![06.jpg](/assets/images/2020-05-13-Directory-Manager/06.jpg)

4. SIZE

    타겟 디렉터리 내 파일의 상대 경로 및 크기를 출력하는 명령어이다.

    - 명령어

        ```bash
        $ SIZE [FILENAME] [OPTION]
        ```

        - FILENAME: 확인할 파일의 절대/상대 경로
        - OPTION
            - -d [NUMBER] : NUMBER 단계 만큼의 하위 디렉터리까지 출력
    - 실행 예시

        ![07.jpg](/assets/images/2020-05-13-Directory-Manager/07.jpg)

        ![08.jpg](/assets/images/2020-05-13-Directory-Manager/08.jpg)

5. RECOVER

    trash 디렉터리에 보관된 파일을 타겟 디렉터리 내 원래 경로로 복원하는 명령어이다.

    - 명령어

        ```bash
        $ RECOVER [FILENAME] [OPTION]
        ```

        - FILENAME: 복구할 파일의 절대/상대 경로
        - OPTION
            - -l : trash 디렉터리에 보관된 파일들을 삭제 시각이 오래된 순으로 출력 후 명령어 수행
    - 실행 예

        ![09.jpg](/assets/images/2020-05-13-Directory-Manager/09.jpg)

        ![10.jpg](/assets/images/2020-05-13-Directory-Manager/10.jpg)

6. TREE

    타겟 디렉터리의 구조를 tree 형태로 출력하는 명령어이다.

    - 명령어

        ```bash
        $ TREE
        ```

    - 실행 예시

        ![11.jpg](/assets/images/2020-05-13-Directory-Manager/11.jpg)

7. EXIT

    dir_manager 프로그램을 종료하는 명령어이다.

    - 명령어

        ```bash
        $ EXIT
        ```

    - 실행 예시 (데몬 프로세스 및 모든 자식 프로세스 종료)

        ![12.jpg](/assets/images/2020-05-13-Directory-Manager/12.jpg)

        ![13.jpg](/assets/images/2020-05-13-Directory-Manager/13.jpg)

8. HELP

    dir_manager의 명령어들의 사용법을 출력하는 명령어이다. 잘못된 명령어를 입력했을 시에도 수행된다.

    - 명령어

        ```bash
        $ HELP
        ```

    - 실행 예시

        ![14.jpg](/assets/images/2020-05-13-Directory-Manager/14.jpg)

## 구현 방법

UNIX의 디렉터리 구조는 Tree 형태로 이루어져 있기 때문에, Tree 자료 구조를 구현했다. 자기 참조형 구조체를 정의해 Node로 사용했다. 각 Node는 Parent와 Child로 이동할 수 있는 자기참조 변수를 갖는데, 이 때 Child의 개수는 정해져 있지 않다. 따라서 자식 Node를 저장하는 포인터 변수는 2차원 포인터로 생성해, 동적 할당을 사용했다. 이렇게 생성한 Tree를 데몬 프로세스가 매 초 마다 새로운 Tree를 생성해 이전에 생성된 Tree와 비교하며 변경사항을 찾아낸다. 또한 DELETE/RECOVER 시 타겟 디렉터리 및 trash 디렉터리에서의 중복 파일 탐색, SIZE 명령어에서의 -d옵션, TREE 명령어에서 디렉터리 계층 구조 출력 등에서도 모두 직접 구현한 Tree를 사용했다.

DELETE 명령어에서는 fork()를 통해 자식 프로세스를 생성한다. 만약 동일한 파일에 대해서 여러 번 DELETE가 예약되었을 경우 그 중 가장 빠른 시각에 삭제되고, 이후에는 무시되도록 하기 위해 Linked List를 생성해 삭제 예정 파일 리스트를 관리했다. fork()를 통해 생성된 자식 프로세스들은 매 초마다 예약 시각에 도달했는지 판단하고, 예약 시각에 도달했을 경우 exit()되어 부모 프로세스에게 SIGCHLD 시그널을 송신한다. 부모 프로세스의 SIGCHILD 시그널 핸들러 함수에서는 DELETE Linked LIst에서 가장 앞의 Node의 파일(가장 빠른 시각의 파일)을 제거한다.

```c
//파일 정보 저장하는 Tree Node
typedef struct FileNode{
    char name[NAME_LEN];
    int childCnt;
    struct FileNode * parent;
    struct FileNode ** child;
    time_t mtime;
    off_t size;
}FileNode;

//시간 정보 저장하는 구조체
typedef struct{
    int year;
    int month;
    int day;
    int hour;
    int min;
    int sec;
}customTime;
 
//DELETE 목록 Linked List의 Node
typedef struct Node {
    char path[PATH_LEN];
    customTime ct;
    bool iOption;
    bool rOption;
    struct Node *prev;
    struct Node *next;
}Node;
```
