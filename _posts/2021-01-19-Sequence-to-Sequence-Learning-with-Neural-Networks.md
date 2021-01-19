---

title: [NLP 논문 리뷰] Sequence To Sequence Learning With Neural Networks (Seq2Seq)
subtitle: Seq2Seq
category: Paper Review
tags: NLP
date: 2021-01-19 12:59:59 +0000
last_modified_at: 2020-09-21 15:14:00 +0000

---

## Paper Info

[Archive Link](https://arxiv.org/abs/1409.3215)

[Paper Link](https://arxiv.org/pdf/1409.3215.pdf)

Submit Date: Sep 10, 2014

---

# Introduction

DNN (Deep Neural Network)는 음성 인식, 사물 인식 등에서 꾸준한 성과를 내어왔다. 하지만 input size가 fixed된다는 한계점이 존재하기 때문에 sequencial problem을 제대로 해결할 수 없다는 한계점이 존재했다. 본 논문에서는 2개의 LSTM (Long Short Term Memory)을 각각 Encoder, Decoder로 사용해 sequencial problem을 해결하고자 했다. 이를 통해 많은 성능 향상을 이루어냈으며, 특히나 long sentence에서 더 큰 상승 폭을 보였다. 이에 더해 단어를 역순으로 배치하는 방식으로도 성능을 향상시켰다.

# The model

![Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20291498ef530149d190ef2b186d28d51f/Untitled.png](/assets/images/2021-01-19-Sequence-to-Sequence-Learning-with-Neural-Networks/Untitled.png)

$$h_t = sigmoid\left(W^{hx}x_t + W^{hh}h_{t-1}\right)\\y_t = W^{yh}h_t$$

$$p\left(y_1,\cdots,y_{T'}|x_1,\cdots,x_T\right)=\prod_{t=1}^{T'}p\left(y_t|v,y_1,\cdots,y_{t-1}\right)$$

RNN은 기본적으로 sequencial problem에 매우 적절한 model이다. 하지만 input size와 output size가 다른 경우에 대해서는 좋은 성능을 보일 수 없었다. 본 논문에서 제시하는 model은 Encoder LSTM에서 하나의 context vector를 생성한 뒤 Decoder LSTM에서 context vector를 이용해 output sentence를 생성하는 방식으로 RNN의 한계점을 극복하고자 했다. input과 output sentence 간의 mapping을 하는 것이 아닌, input sentence를 통해 encoder에서 context vector를 생성하고, 이를 활용해 decoder에서 output sentence를 만들어내는 것이다. Encoder LSTM의 output인 context vector는 Encoder의 마지막 layer에서 나온 output이다. 이를 Decoder LSTM의 첫번째 layer의 input으로 넣게 된다. 여기서 주목할만한 점은 input sentence에서의 word order를 reverse해 사용했다는 것이다. 또한 <EOS> (End of Sentence) token을 각 sentence의 끝에 추가해 variable length sentence를 다뤘다.

## Experiments

WMT'14의 English to French dataset으로 실험을 진행했다. source / target language 각각에 fixed size vocabulary를 사용했다 (source: 160,000 / target: 80,000). OOV는 "UNK" token으로 대체된다. long sequence에서는 source sentence를 reverse시킨 경우가 특히나 성능이 더 좋았다. 구체적인 수치로 BLEU score가 25.9에서 30.6으로 증가했다.

- 원래의 순서대로 나열된 단어의 경우 source와 target에서의 연결되는 단어쌍(pair of word) 사이의 거리가 모두 동일하다. 하지만 reverse시킬 경우에는 sentence에서 앞에 위치한 단어일수록 단어쌍 사이의 거리가 짧아지게 된다. 이는 결국 sentence에서 뒤에 위치한 단어들에 대해서는 오히려 reverse하지 않았을 때보다 단어쌍 사이의 거리가 멀어지는 결과를 낳는다. 생각해보면 결국 reverse한 뒤나, 하지 않았을 때에나 단어쌍 사이의 거리 mean값은 동일하다. 그런데 왜 reverse시 더 좋은 성능이 나오는 것인지 의문일 수 있는데, sequencial problem의 개념으로 다시 돌아와 생각해보면 어느정도 이유를 추론 가능하다. sequencial problem에서는 앞쪽에 위치한 data가 뒤의 모든 data에 영향을 주기 때문에 앞에 위치한 data일 수록 중요도가 더 높다고 할 수 있다. 따라서 reverse를 통해 source sentence에서 앞쪽에 위치한 data(word)들의 target sentence에서의 연관 word와의 거리를 줄이는 것은 더 중요도 높은 data에 대해 더 좋은 성능을 보장하게 되는 효과를 낳는다.

dataset의 대부분은 short length sentence이기에 mini batch 사용 시 각 batch 마다 아주 적은 수의 long length sentence가 포함되는 문제가 존재했다. 따라서 각 batch마다 대략적으로 비슷한 length를 가진 sentence가 포함되도록 normalization을 수행한 뒤 실험을 진행했다.

![Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20291498ef530149d190ef2b186d28d51f/05-16-2020-17.33.39.jpg](/assets/images/2021-01-19-Sequence-to-Sequence-Learning-with-Neural-Networks/05-16-2020-17.33.39.jpg)

![Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20291498ef530149d190ef2b186d28d51f/05-16-2020-17.33.54.jpg](/assets/images/2021-01-19-Sequence-to-Sequence-Learning-with-Neural-Networks/05-16-2020-17.33.54.jpg)

SOTA(State of the Art)에 비해 0.5 낮은 BLEU Score를 달성했다. OOV가 여전히 존재함에도 SOTA와 동등한 성능을 달성했다는 것은 충분히 의미가 있다.

위에서 언급했듯이 long Sentence에서도 매우 좋은 성능을 보였다.

![Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20291498ef530149d190ef2b186d28d51f/05-16-2020-17.39.20.jpg](/assets/images/2021-01-19-Sequence-to-Sequence-Learning-with-Neural-Networks/05-16-2020-17.39.20.jpg)

![Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20291498ef530149d190ef2b186d28d51f/05-16-2020-17.42.43.jpg](/assets/images/2021-01-19-Sequence-to-Sequence-Learning-with-Neural-Networks/05-16-2020-17.42.43.jpg)
