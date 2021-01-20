---
title: sync manager
subtitle: [System Programming] 동기화 프로그램 구현
categories: System Programming
tags: System Programming
date: 2021-01-20 19:22:46 +0000
last_modified_at: 2021-01-20 19:22:46 +0000
---

Env: Ubuntu 18.04 LTS, gcc 7.5.0
GitHub: https://github.com/cpm0722/LSP/tree/master/rsync

```yaml
cleanUrl: /lsp/sync-manager
disqus: true
```

## 개요

파일 및 디렉터리 간 동기화를 수행하는 프로그램이다. 동기화시킬 대상인 src는 파일, 디렉터리 모두 가능하지만, 동기화될 대상인 dst는 디렉터리만 가능하다. 만약 dst 디렉터리 내에 이미 src 내의 파일과 동일한 파일이 있을 경우 동기화를 수행하지 않는다. 동기화가 완료될 경우 log.txt파일에 동기화 과정에서 추가된 파일, 삭제된 파일이 기록된다.

## 실행 방법

```bash
$ make rsync
./rsync [SOURCE_FILE_OR_DIRECTORY_PATH] [DESTINATION_DIRECTORY_PATH] [OPTION]
```

- SOURCE_FILE_OR_DIRECTORY_PATH: 동기화시킬 대상인 파일 또는 디렉터리의 절대/상대 경로
- DESTINATION_DIRECTORY_PATH: 동기화될 대상인 디렉터리의 절대/상대 경로
- OPTION
    - -r : src 디렉터리의 서브 디렉터리 내의 파일에 대해서도 동기화 수행
    - -m : src 디렉터리 내에 존재하지 않는 파일 또는 디렉터리가 dst 디렉터리에 존재할 경우 삭제
    - -t : 동기화 대상들을 tar을 이용해 압축해 dst 디렉터리로 전송 후 압축 해제

## 기능

1. 파일 동기화

    ![sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.33.24.jpg](sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.33.24.jpg)

2. -m 옵션 사용한 동기화

    ![sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.34.42.jpg](sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.34.42.jpg)

3. -r 옵션 사용한 디렉터리 동기화

    ![sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.38.35.jpg](sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.38.35.jpg)

4. -t 옵션 사용한 압축 동기화

    ![sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.39.57.jpg](sync%20manager%20cb99b96ddea344bbbabcb1f58d2c386a/10-04-2020-22.39.57.jpg)

## 구현 방법

두 파일이 동일한 파일인지는 아래의 3가지 기준으로 판단한다.

1. 파일명
2. 파일 크기
3. 최종 수정 시각

두 파일에 대한 경로를 인자로 받아 같은 파일인지를 return하는 is_same_file 함수를 정의해 사용했다. -r 옵션을 위해 동기화 함수를 재귀적으로 호출 가능하도록 작성했다. -t 옵션에서는 압축 및 해제를 수행하는데, system() 함수를 사용해 tar 명령어를 호출했다.

## 소스 코드

[cpm0722/LSP](https://github.com/cpm0722/LSP/tree/master/rsync)