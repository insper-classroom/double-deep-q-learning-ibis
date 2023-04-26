[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/SnaQZIS-)



**Reinforcement Learning - Questionário Projeto 1**


Alunos: Diogo Duarte, Eduardo Araujo e Felipe Schiavinato


  **1. Qual é a característica do Q-Learning e Deep Q-Learning destacada pelos autores do artigo?**

  A principal característica destacada por esses dois algoritmos pelos autores é a superestimação de valores, e que isso pode levar a políticas de agentes piores, além de uma maior instabilidade no treinamento dos agentes, contudo, vale ressaltar que nem sempre a superestimação leva a agentes piores desde que elas sejam uniformes e bem distribuídas entre os estados de um enviroment. 


  **2. Qual é a principal ideia do Double Deep Q-Learning e como esta ideia deve ser implementada? Mostre a ideia através de um pseudo-código.**
  
  A principal ideia do Double Deep Q-Learning é utilizar duas redes para selecionar e avaliar as ações separadamente, reduzindo as superestimações


  **3. Como os testes empíricos foram implementados neste artigo?**
  
  
  Os testes empíricos foram realizados em jogos do Atari 2600 através do Arcade Learning Environment, um framework em Python que contempla jogos do Atari 2600. A implementação foi feita de uma maneira que um único algoritmo com um conjunto fixo de hiperparâmetros aprenderia jogar diferentes jogos desse framework, então, foi feita a comparação das políticas desenvolvidas pelo DDQN (Double Deep Q-Learning Network) com as feitas pelo DQN (Deep Q-Learning).
Por fim, as configurações das redes neurais escolhidas foram as mesmas propostas por Volodymyr Mnihz, a rede utiliza os últimos 4 frames do jogo como inputs, e os  outputs são os valores para cada ação possível.


  **4. Quais foram as principais contribuições do artigo?**
  
Dentre as contribuições do artigo estão, mostrar que o algoritmo Q-Learning é excessivamente optimista em alguns cenários e que isso é mais comum e frequente do que antes se imaginava. Além disso, o artigo evidenciou que o Double Deep Q-Learning pode ser usado para reduzir esse excesso de optimismo e consequentemente desenvolver agentes mais estáveis e com políticas melhores. Por fim, foi proposto uma implementação para o Double Deep Q-Learning usando como referência uma configuração já existente de uma implementação de Deep Q-Learning.

  **5. Como é que podemos verificar que o Double Deep Q-Learning tem um aprendizado mais estável e consegue encontrar políticas melhores?**
	Para realizar essas análises, além de uma análise matemática, foi feita uma comparação usando os testes empíricos nos jogos de Atari. Para o aprendizado estável, é possível analisar as curvas de Score X Training Steps, e verificar o quão instável são essas curvas durante o treinamento dos agentes. E realmente, analisando as curvas para os jogos Wizard of Wor e Asterix, o Double Deep Q-Learning apresentou uma estabilidade maior em relação ao Deep Q-Learning. Para avaliar a qualidade das políticas, pode-se avaliar nos gráficos de Value Estimates X Training Steps, os true values de cada um dos algoritmos, e dos quatro jogos apresentados, os true values do Double Deep Q-Learning foram melhores do que os do Deep Q-Learning, indicando que as políticas deste são melhores.

**6. Implemente o algoritmo Double Deep Q-Learning e valide nos seguintes ambientes: Cart Pole e Lunar Lander.**

**7. Você conseguiu o comportamento esperado? Justifique a sua resposta. Em caso negativo, na opinião do grupo, o que faltou implementar para conseguir o comportamento esperado?**

Sim, a implementação do DDQN apresentou resultados melhores que os da implementação DQN, podendo ser visto esses resultados nos gráficos.
