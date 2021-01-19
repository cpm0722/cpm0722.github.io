var store = [{
        "title": "[NLP 논문 리뷰] Neural Machine Translation of Rare Words with Subword Units (BPE)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Aug 15, 2015 Backgrounds BLEU Score (Bilingual Evaluation Understudy) score \\[BLEU=min\\left(1,\\frac{\\text{output length}}{\\text{reference_length}}\\right)\\left(\\prod_{i=1}^4precision_i\\right)^{\\frac{1}{4}}\\] reference sentence와 output sentence의 일치율을 나타내는 score이다. 3단계 절차를 거쳐 최종 BLEU Score를 도출해낸다. n-gram에서 순서쌍의 겹치는 정도 (Precision) Example output sentence 빛이 쐬는 노인은 완벽한 어두운 곳에서 잠든 사람과...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Neural-Machine-Translation-of-Rare-Words-with-Subword-Units/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Sequence To Sequence Learning With Neural Networks (Seq2Seq)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Sep 10, 2014 Introduction DNN (Deep Neural Network)는 음성 인식, 사물 인식 등에서 꾸준한 성과를 내어왔다. 하지만 input size가 fixed된다는 한계점이 존재하기 때문에 sequencial problem을 제대로 해결할 수 없다는 한계점이 존재했다. 본 논문에서는 2개의 LSTM (Long Short Term Memory)을 각각 Encoder, Decoder로...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Sequence-to-Sequence-Learning-with-Neural-Networks/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Neural Machine Translation By Jointly Learning To Align And Translate (Attention Seq2Seq)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Sep 1, 2014 Abstract 기존의 seq2seq model에서 사용된 LSTM을 사용한 encoder-decoder model은 sequential problem에서 뛰어난 성능을 보였다. 하지만 encoder에서 생성해낸 context vector를 decoder에서 sentence로 만들어내는 위와 같은 방식에서 고정된 vector size는 긴 length의 sentence에서 bottleneck이 된다는 사실을 발견했다. 본 논문에서는 이러한 문제점을...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Neural-Machine-Translation-By-Jointly-Learning-To-Align-And-Translate/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Attention Is All You Need (Transformer)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 12, 2017 Introduction RNN과 LSTM을 사용한 Neural Network 접근 방식은 Sequencial Transduction Problem에서 매우 좋은 성능을 달성했다. 그 중 특히 Encoder-Decoder를 사용한 Attention Model이 state-of-art를 달성했다. 하지만 Recurrent Model은 본질적으로 한계가 존재하는데, 바로 Parallelization이 불가능하다는 문제점이다. 이는 Sequence의 길이가 긴 상황에서...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Attention-is-All-You-Need/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Oct 11, 2018 Introduction NLP에서도 pre-trained된 model을 사용하는 기법은 있었지만, pre-train에서 단방향의 architecture만 사용할 수 있다는 한계점이 있었다. 이는 양방향에서의 문맥 정보가 모두 중요한 token-level task에서 좋은 성능을 보이지 못하는 원인이 되었다. 본 논문에서는 MLM(Masked Language Model)을 사용해 bidirectional한 context도 담을 수...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Xlnet: Generalized Autoregressive Pretraining for Language Understanding",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 19, 2019 Introduction Unsupervised Learning을 pretraining에 적용시키는 방식은 NLP domain에서 매우 큰 성과를 이뤄냈다. Unsupervised pretraining하는 방법론은 크게 AutoRegressive(AR)과 AutoEncoding(AE)가 있다. AutoRegressive는 순방향 또는 역방향으로 다음의 corpus를 예측하는 방식으로 학습한다. 이는 단방향 context만 학습할 수 있다는 단점이 있다. 하지만 현실의 대부분의...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/XLNet-Generalized-Autoregressive-Pretraining-for-Language-Understanding/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] MASS: Masked Sequence To Sequence Pre Training For Language Generation",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 21, 2019 Abstract BERT에서 영감을 받아 Pre-training / fine-tuning, encoder/decoder를 채택한 MAsked Sequence to Sequence (MASS) model을 만들어냈다. random하게 input sentence에 연속적인 mask를 부여한 뒤 decoder가 predict하는 방식으로 encoder와 decoder를 Pre-training시켜 Language Generation Task에 적합한 Model을 개발했다. 특히 dataset이 적은 Language...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/MASS-Masked-Sequence-to-Sequence-Pre-training-for-Language-Generation/",
        "teaser": null
      },{
        "title": "[운영체제] Introduction to Operating Systems",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. OS (Operating System) OS의 정의 OS는 SW의 일종이다. ‘자원’(HW)관리’자’(SW)로 정의할 수 있다. HW를 SW로 관리해주는 역할을 수행한다. 구체적으로 CPU, Memory, Disk 등을 struct로 정의해 각각 Process, Virtual Memory, File System을 만들어낸다. OS는 kernel 함수를 이용해 HW를 관리하며, 사용자는 kernel 함수를...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Introduction-to-Operating-Systems/",
        "teaser": null
      },{
        "title": "[운영체제] Process Abstraction",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Process / Program process와 program를 우선 정의내려 보자. program은 disk에 위치한다. 반면 process는 memory에 위치한다. process는 disk에 위치한 program file을 memory에 올린 것이다. 이 때 program file 전체를 모두 memory에 올릴 수도, 필요한 일부분만 memory에 올릴 수도 있다. 정리하자면, process는...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Process-Abstraction/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Subword-level Word Vector Representation for Korean",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jul 1, 2018 Abstract 지금까지의 word representation에 관한 연구는 모두 영어에 집중되어 왔다. language-specific knowledge는 언어 학습에 있어서 매우 결정적인 요소인데 영어와 기원이 다른 여러 다양한 언어들에 대해서는 이러한 연구가 부진했던 것이 사실이다. 본 논문에서는 한국어만의 unique한 언어 구조를 분석해 NLP에 적용해보고자...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/Subword-level-Word-Vector-Representation-for-Korean/",
        "teaser": null
      },{
        "title": "[운영체제] Scheduling",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. 단기 Scheduling Scheduling에는 여러 종류가 있다. 장기(long-term) scheduling은 process가 CPU에 의해 실행될 자격을 부여할 지를 결정하는 것이다. 중기(medium-term) scheduling은 process(의 일부)가 Memory에 올라갈 자격을 부여할 지를 결정하는 것이다. 단기(short-term) scheduling은 CPU에 실행될 다음 process를 선택하는 것으로, Dispatcher라고 불린다. 아래에서는 단기...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Scheduling/",
        "teaser": null
      },{
        "title": "[운영체제] Scheduling: MLFQ(Multi Level Feedback Queue)",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. RR과 MLFQ Round Robin 기법은 평균 응답 시간은 최소화시켰지만, 평균 반환 시간은 최악이라는 점에서 한계가 있었다. 물론 평균 응답 시간이 짧기 때문에 사용자가 속도가 빠른 시스템으로 인지하도록 착각을 유도할 수 있었다. 하지만 짧은 task에 대해서도 slicing을 수행해 여러 번 나눠...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Scheduling-MLFQ/",
        "teaser": null
      },{
        "title": "[운영체제] Address & Memory",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Virtual Memory의 등장 과정 Single Programming 초기 (1950~1970)의 운영체제는 물리 memory에 하나의 program만을 올리는 Single Programming 형태였다. 즉, memory는 OS 영역, 실행 중인 1개의 program이 올라가는 영역으로 구분됐다. memory 가상화에 대한 개념도 존재하지 않았다. 동시에 memory를 점유할 수 있는 program의...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Address-Memory/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jul 26, 2019 Introduction BERT 계열의 model들은 지금까지 매우 뛰어난 성능을 보여왔다. 본 논문에서는 BERT에 대한 추가적인 연구를 통해 기존의 BERT model들이 undertrained되었음을 보여주고, 다음의 개선 방안들을 제시한다. 더 긴 시간, 더 큰 batch size의 training NSP 제거 long sequence에 대한 학습...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/RoBERTa-A-Robustly-Optimized-BERT-Pretraining-Approach/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Oct 6, 2020 Introduction NLP에서 Tokenization은 전처리 과정에서 가장 중요한 issue 중 하나이다. 가장 적절한 Tokenization 전략을 찾기 위한 연구는 수도 없이 이루어져 왔다. 그 중 가장 대표적인 방식이 BPE이다. BPE는 많은 연구를 통해 보편적으로 가장 효율적인 Tokenization 기법으로 알려졌지만, 아직 language나...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/An-Empirical-Study-of-Tokenization-Strategies-for-Various-Korean-NLP-Tasks/",
        "teaser": null
      },{
        "title": "[운영체제] Paging Mechanism",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Paging Mechanism paging 기법에 대해 자세히 알아보자. 위에서 살펴본 고정 분할 및 가변 분할 기법은 각각 내부 단편화, 외부 단편화의 문제점이 존재했다. paging은 이러한 단점들을 해결하기 위해 고안된 방식이다. paging을 사용하면 결론적으로 외부 단편화는 발생하지 않으며, 내부 단편화는 아주 적은...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Paging-Mechanism/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] KR-BERT: A Small Scale Korean Specific Language Model",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Aug 10, 2020 Introduction 기존의 BERT model은 104개의 language의 Wikipedia dataset으로 학습된 model이다. 범용적으로 사용될 수 있다는 장점에도 불구하고, model의 크기가 과도하게 크다는 단점이 존재한다. 또한 non-English downstream task에서 좋은 성능을 보여주지 못하는 경우가 많다는 한계도 명확하다. 특히나 Korean과 같은 언어에서는 한계가...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/KR-BERT-A-Small-Scale-Korean-Specific-Language-Model/",
        "teaser": null
      },{
        "title": "[운영체제] Concurrency",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Thread Process OS에서 process는 역할을 정리해보자. 우선 process는 자원 소유의 단위이다. 자원이라는 것은 main memory, I/O device, file system 등을 의미한다. 대표적인 예시로 process별로 main memory에 서로 다른 공간을 할당하는 것이 있다. 두번째로 process는 scheduling의 단위이다. context switching은 process 사이에...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Concurrency/",
        "teaser": null
      },{
        "title": "[운영체제] Semaphore",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Semaphore semaphore는 다수의 thread 사이의 병행성 유지를 위해 OS 단위에서 제공되는 기법이다. 기본적인 작동 원리는 특정 thread가 특정 signal을 수신할 때까지 정해진 위치에서 wait하도록 강제하는 것이다. counting semaphore counting semahpore는 정수값을 갖는 counting 변수와 3가지 연산으로 구성된다. 범용 semaphore라고도 불리운다....","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Semaphore/",
        "teaser": null
      },{
        "title": "[운영체제] File System",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Block OS는 disk를 일정한 크기의 block으로 나누어 저장한다. 대개 block의 크기는 4KB이다. 각 block은 그 목적에 따라 아래의 4가지로 구분지을 수 있다. Super block file system의 global한 정보들을 담는 block으로 하나의 file system에 1개만 존재한다. Allocation structure block bitmap, linked...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/File-System/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Efficient Estimation Of Word Representations In Vector Space (Word2Vec)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jan 16, 2013 Introduction one-hot encoding 방식은 word를 단순하게 표현하는 방법이다. word 자체가 갖는 정보를 담고 있지 않고 단순하게 index만을 담고 있는데, index 역시 word에 내재된 어떤 정보와도 관련이 없다. 본 논문에서는 word vector에 word 자체가 담고 있는 의미를 확실하게 담아내고자 했다....","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Efficient-Estimation-of-Word-Representations-in-Vector-Space/",
        "teaser": null
      },{
        "title": "[운영체제] Disk",
        "excerpt":"숭실대학교 컴퓨터학부 홍지만 교수님의 2020-2학기 운영체제 강의를 정리 및 재구성했다. Hard Disk hard disk는 가장 범용적으로 사용되는 저장 장치이다. main memory와 다르게 영속적(persistent)으로 data를 저장할 수 있다. hard disk는 물리적으로 회전(rotation)하면서 data를 저장할 장소를 찾는다. 전체 구성 요소는 다음과 같다. hard disk는 여러 층으로 이루어져 있다. 각 층은 platter라는 하나의...","categories": ["Operating System"],
        "tags": ["Operating System"],
        "url": "http://0.0.0.0:4000/operating%20system/Disk/",
        "teaser": null
      },{
        "title": "[NLP 논문 리뷰] Deep Contextualized Word Representations (ELMo)",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Feb 15, 2018 Introduction word2vec이나 glove와 같은 기존의 word embedding 방식은 다의어의 모든 의미를 담아내기 곤란하다는 심각한 한계점을 갖고 있다. ELMo(Embeddings from Language Models)는 이러한 한계점을 극복하기 위해 embedding에 sentence의 전체 context를 담도록 했다. pre-train된 LSTM layer에 sentence 전체를 넣어 각 word의...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Deep-contextualized-word-representations/",
        "teaser": null
      }]
