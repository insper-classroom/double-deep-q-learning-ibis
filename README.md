[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/SnaQZIS-)



## Reinforcement Learning


**Alunos: Diogo Duarte, Eduardo Araujo e Felipe Schiavinato**

**Data: 27/04/2023**


### Sobre o projeto

A intenção do projeto é o estudo e a implementação do algoritmo Double Deep Q-Learning para o treinamento de agentes em alguns ambientes. Para isso, utilizou-se como base o artigo publicado por van Hasselt, Guez e Silver [1].



### Questionário do projeto


**1. Qual é a característica do Q-Learning e Deep Q-Learning destacada pelos autores do artigo?**

A principal característica destacada por esses dois algoritmos pelos autores é a superestimação de valores, e que isso pode levar a políticas de agentes piores, além de uma maior instabilidade no treinamento dos agentes, contudo, vale ressaltar que nem sempre a superestimação leva a agentes piores desde que elas sejam uniformes e bem distribuídas entre os estados de um enviroment. 


**2. Qual é a principal ideia do Double Deep Q-Learning e como esta ideia deve ser implementada? Mostre a ideia através de um pseudo-código.**
  
A principal ideia do Double Deep Q-Learning é utilizar duas redes para selecionar e avaliar as ações separadamente, reduzindo as superestimações
  
    Cria-se duas redes neurais inicialmente iguais.
    Para cada N episódios:
          Para cada passo:
                Utiliza uma das redes para escolher a ação e outra para avaliar as ações.

                chama a função de loss e atualiza os pesos da rede de seleção.
                
          A cada X episódios:
                Iguala weights das rede de avaliação com os da rede de seleção.



**3. Como os testes empíricos foram implementados neste artigo?**
  
  
Os testes empíricos foram realizados em jogos do Atari 2600 através do Arcade Learning Environment, um framework em Python que contempla jogos do Atari 2600. A implementação foi feita de uma maneira que um único algoritmo com um conjunto fixo de hiperparâmetros aprenderia jogar diferentes jogos desse framework, então, foi feita a comparação das políticas desenvolvidas pelo DDQN (Double Deep Q-Learning Network) com as feitas pelo DQN (Deep Q-Learning).
Por fim, as configurações das redes neurais escolhidas foram as mesmas propostas por Volodymyr Mnihz, a rede utiliza os últimos 4 frames do jogo como inputs, e os  outputs são os valores para cada ação possível.


**4. Quais foram as principais contribuições do artigo?**
  
Dentre as contribuições do artigo estão, mostrar que o algoritmo Q-Learning é excessivamente optimista em alguns cenários e que isso é mais comum e frequente do que antes se imaginava. Além disso, o artigo evidenciou que o Double Deep Q-Learning pode ser usado para reduzir esse excesso de optimismo e consequentemente desenvolver agentes mais estáveis e com políticas melhores. Por fim, foi proposto uma implementação para o Double Deep Q-Learning usando como referência uma configuração já existente de uma implementação de Deep Q-Learning.

**5. Como é que podemos verificar que o Double Deep Q-Learning tem um aprendizado mais estável e consegue encontrar políticas melhores?**
  
Para realizar essas análises, além de uma análise matemática, foi feita uma comparação usando os testes empíricos nos jogos de Atari. Para o aprendizado estável, é possível analisar as curvas de Score X Training Steps, e verificar o quão instável são essas curvas durante o treinamento dos agentes. E realmente, analisando as curvas para os jogos Wizard of Wor e Asterix, o Double Deep Q-Learning apresentou uma estabilidade maior em relação ao Deep Q-Learning. Para avaliar a qualidade das políticas, pode-se avaliar nos gráficos de Value Estimates X Training Steps, os true values de cada um dos algoritmos, e dos quatro jogos apresentados, os true values do Double Deep Q-Learning foram melhores do que os do Deep Q-Learning, indicando que as políticas deste são melhores.

**6. Implemente o algoritmo Double Deep Q-Learning e valide nos seguintes ambientes: Cart Pole e Lunar Lander.**

Primeiramente, foi feita a implementação do algoritmo Double Deep Q-Learning, e para isso, foi utilizado como base a implementação do Deep Q-Learning feita em sala de aula. A implementação do Double Deep Q-Learning pode ser vista no arquivo `agents.py`. Para validar o algoritmo, foram utilizados os ambientes Cart Pole e Lunar Lander, e os resultados podem ser vistos nos gráficos abaixo.

![](imgs/DDQN_vs_DQN.png)
![](imgs/DDQN_vs_DQN_stats.png)

Em seguida o treinamento do Cart Pole:

![](imgs/DDQL_CartPole-v1.png)


Comparando com DQN (Deep Q-Learning), o DDQN (Double Deep Q-Learning) apresentou resultados melhores, tanto no Cart Pole quanto no Lunar Lander, mas não tão superiores quanto ao desempenho aumentando na média e no maximo e diminuindo no mínimo, talvez isso se deva ao treinamento contar apenas com 1000 episodios. Como visto no artigo o DQN tende a diminuir seus rewards apos um periodo enquanto o DDQL tende a mante-los mesmo apos o reward maximo durante o aprendizado. Uma caracteristica que podemos observar na comparação do Lunar Lander é que o o DDQN é mais estavel durante o aprendizado.

Finalmente, foi implementado uma rede neural convolucional para o DDQN, visando treinar ambientes do Atari. Infelizmente, não foi possível realizar o treinamento devido a limitações de hardware, mas a implementação também pode ser vista no arquivo `agents.py`.
Além disso, uma tentativa de treinamento pode ser vista abaixo.

![](DDQL_BreakoutNoFrameskip-v4.png)

**7. Você conseguiu o comportamento esperado? Justifique a sua resposta. Em caso negativo, na opinião do grupo, o que faltou implementar para conseguir o comportamento esperado?**

Sim, a implementação do DDQN apresentou resultados melhores que os da implementação DQN, podendo ser visto esses resultados nos gráficos.

### Referências

1. van Hasselt, H., Guez, A. and Silver, D. 2016. Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence. 30, 1 (Mar. 2016). DOI: https://doi.org/10.1609/aaai.v30i1.10295.
