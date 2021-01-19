var store = [{
        "title": "test title v1",
        "excerpt":"\\[\\sum^N_{i=1}i\\] ","categories": [],
        "tags": [],
        "url": "http://0.0.0.0:4000/test/",
        "teaser": null
      },{
        "title": "Neural Machine Translation of Rare Words with Subword Units",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Aug 15, 2015 Backgrounds BLEU Score (Bilingual Evaluation Understudy) score \\[BLEU=min\\left(1,\\frac{\\text{output length}}{\\text{reference_length}}\\right)\\left(\\prod_{i=1}^4precision_i\\right)^{\\frac{1}{4}}\\] reference sentence와 output sentence의 일치율을 나타내는 score이다. 3단계 절차를 거쳐 최종 BLEU Score를 도출해낸다. n-gram에서 순서쌍의 겹치는 정도 (Precision) Example output sentence 빛이 쐬는 노인은 완벽한 어두운 곳에서 잠든 사람과...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Neural-Machine-Translation-of-Rare-Words-with-Subword-Units/",
        "teaser": null
      },{
        "title": "An Empirical Study Of Tokenization Strategies For Various Korean Nlp Tasks",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Oct 6, 2020 Introduction NLP에서 Tokenization은 전처리 과정에서 가장 중요한 issue 중 하나이다. 가장 적절한 Tokenization 전략을 찾기 위한 연구는 수도 없이 이루어져 왔다. 그 중 가장 대표적인 방식이 BPE이다. BPE는 많은 연구를 통해 보편적으로 가장 효율적인 Tokenization 기법으로 알려졌지만, 아직 language나...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/An-Empirical-Study-of-Tokenization-Strategies-for-Various-Korean-NLP-Tasks/",
        "teaser": null
      },{
        "title": "Mass Masked Sequence To Sequence Pre Training For Language Generation",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 21, 2019 Abstract BERT에서 영감을 받아 Pre-training / fine-tuning, encoder/decoder를 채택한 MAsked Sequence to Sequence (MASS) model을 만들어냈다. random하게 input sentence에 연속적인 mask를 부여한 뒤 decoder가 predict하는 방식으로 encoder와 decoder를 Pre-training시켜 Language Generation Task에 적합한 Model을 개발했다. 특히 dataset이 적은 Language...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/MASS-Masked-Sequence-to-Sequence-Pre-training-for-Language-Generation/",
        "teaser": null
      },{
        "title": "Xlnet Generalized Autoregressive Pretraining For Language Understanding",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 19, 2019 Introduction Unsupervised Learning을 pretraining에 적용시키는 방식은 NLP domain에서 매우 큰 성과를 이뤄냈다. Unsupervised pretraining하는 방법론은 크게 AutoRegressive(AR)과 AutoEncoding(AE)가 있다. AutoRegressive는 순방향 또는 역방향으로 다음의 corpus를 예측하는 방식으로 학습한다. 이는 단방향 context만 학습할 수 있다는 단점이 있다. 하지만 현실의 대부분의...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/XLNet-Generalized-Autoregressive-Pretraining-for-Language-Understanding/",
        "teaser": null
      },{
        "title": "Sequence To Sequence Learning With Neural Networks",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Sep 10, 2014 Introduction DNN (Deep Neural Network)는 음성 인식, 사물 인식 등에서 꾸준한 성과를 내어왔다. 하지만 input size가 fixed된다는 한계점이 존재하기 때문에 sequencial problem을 제대로 해결할 수 없다는 한계점이 존재했다. 본 논문에서는 2개의 LSTM (Long Short Term Memory)을 각각 Encoder, Decoder로...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Sequence-to-Sequence-Learning-with-Neural-Networks/",
        "teaser": null
      },{
        "title": "Bert Pre Training Of Deep Bidirectional Transformers For Language Understanding",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Oct 11, 2018 Introduction NLP에서도 pre-trained된 model을 사용하는 기법은 있었지만, pre-train에서 단방향의 architecture만 사용할 수 있다는 한계점이 있었다. 이는 양방향에서의 문맥 정보가 모두 중요한 token-level task에서 좋은 성능을 보이지 못하는 원인이 되었다. 본 논문에서는 MLM(Masked Language Model)을 사용해 bidirectional한 context도 담을 수...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/",
        "teaser": null
      },{
        "title": "Subword Level Word Vector Representation For Korean",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jul 1, 2018 Abstract 지금까지의 word representation에 관한 연구는 모두 영어에 집중되어 왔다. language-specific knowledge는 언어 학습에 있어서 매우 결정적인 요소인데 영어와 기원이 다른 여러 다양한 언어들에 대해서는 이러한 연구가 부진했던 것이 사실이다. 본 논문에서는 한국어만의 unique한 언어 구조를 분석해 NLP에 적용해보고자...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/Subword-level-Word-Vector-Representation-for-Korean/",
        "teaser": null
      },{
        "title": "Roberta A Robustly Optimized Bert Pretraining Approach",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jul 26, 2019 Introduction BERT 계열의 model들은 지금까지 매우 뛰어난 성능을 보여왔다. 본 논문에서는 BERT에 대한 추가적인 연구를 통해 기존의 BERT model들이 undertrained되었음을 보여주고, 다음의 개선 방안들을 제시한다. 더 긴 시간, 더 큰 batch size의 training NSP 제거 long sequence에 대한 학습...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/RoBERTa-A-Robustly-Optimized-BERT-Pretraining-Approach/",
        "teaser": null
      },{
        "title": "Attention Is All You Need",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jun 12, 2017 Introduction RNN과 LSTM을 사용한 Neural Network 접근 방식은 Sequencial Transduction Problem에서 매우 좋은 성능을 달성했다. 그 중 특히 Encoder-Decoder를 사용한 Attention Model이 state-of-art를 달성했다. 하지만 Recurrent Model은 본질적으로 한계가 존재하는데, 바로 Parallelization이 불가능하다는 문제점이다. 이는 Sequence의 길이가 긴 상황에서...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Attention-is-All-You-Need/",
        "teaser": null
      },{
        "title": "Efficient Estimation Of Word Representations In Vector Space",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Jan 16, 2013 Introduction one-hot encoding 방식은 word를 단순하게 표현하는 방법이다. word 자체가 갖는 정보를 담고 있지 않고 단순하게 index만을 담고 있는데, index 역시 word에 내재된 어떤 정보와도 관련이 없다. 본 논문에서는 word vector에 word 자체가 담고 있는 의미를 확실하게 담아내고자 했다....","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Efficient-Estimation-of-Word-Representations-in-Vector-Space/",
        "teaser": null
      },{
        "title": "Neural Machine Translation By Jointly Learning To Align And Translate",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Sep 1, 2014 Abstract 기존의 seq2seq model에서 사용된 LSTM을 사용한 encoder-decoder model은 sequential problem에서 뛰어난 성능을 보였다. 하지만 encoder에서 생성해낸 context vector를 decoder에서 sentence로 만들어내는 위와 같은 방식에서 고정된 vector size는 긴 length의 sentence에서 bottleneck이 된다는 사실을 발견했다. 본 논문에서는 이러한 문제점을...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Neural-Machine-Translation-By-Jointly-Learning-To-Align-And-Translate/",
        "teaser": null
      },{
        "title": "Deep Contextualized Word Representations",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Feb 15, 2018 Introduction word2vec이나 glove와 같은 기존의 word embedding 방식은 다의어의 모든 의미를 담아내기 곤란하다는 심각한 한계점을 갖고 있다. ELMo(Embeddings from Language Models)는 이러한 한계점을 극복하기 위해 embedding에 sentence의 전체 context를 담도록 했다. pre-train된 LSTM layer에 sentence 전체를 넣어 각 word의...","categories": ["Paper Review"],
        "tags": ["NLP"],
        "url": "http://0.0.0.0:4000/paper%20review/Deep-contextualized-word-representations/",
        "teaser": null
      },{
        "title": "Kr Bert A Small Scale Korean Specific Language Model",
        "excerpt":"Paper Info Archive Link Paper Link Submit Date: Aug 10, 2020 Introduction 기존의 BERT model은 104개의 language의 Wikipedia dataset으로 학습된 model이다. 범용적으로 사용될 수 있다는 장점에도 불구하고, model의 크기가 과도하게 크다는 단점이 존재한다. 또한 non-English downstream task에서 좋은 성능을 보여주지 못하는 경우가 많다는 한계도 명확하다. 특히나 Korean과 같은 언어에서는 한계가...","categories": ["Paper Review"],
        "tags": ["NLP","Korean"],
        "url": "http://0.0.0.0:4000/paper%20review/KR-BERT-A-Small-Scale-Korean-Specific-Language-Model/",
        "teaser": null
      }]
