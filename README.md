# Yumaru 2.0

Esse repositório é uma versão atualizada de um código de processamento de linguagem natural apelidado de Yumaru, a qual tem a mesma finalidade de sua versão antecessora, a de encontrar palavras próximas a um dado input, mas dessa vez utilizando apenas técnicas de vetorização de palavras e manipulação de tensores, sem ter o deep learning presente. Assim, o modelo acaba não tendo a complexidade do aprendizado profundo, porém conseguindo cumprir sua função de maneira mais otimizada em relação ao primeiro projeto.

## Objetivos e métodos
A principal meta desse projeto é a de encontrar relações entre as palvras de textos, de modo que elas estejam numericamente próximas umas das outras. Para isso, utiliza-se primeiramente o word2vec para que cada sentença vire um vetor, podendo assim passar por uma análise mais detalhada. Para o código foram utilizados os seguintes recursos:
### Pytorch
O pytorch, uma biblioteca de manipulação de tensores, teve como função formatar e redimensionar os vetores criados pelo word2vec.
### nltk
A biblioteca nltk oferece uma variedade de dados sobre linguagem natural, que foram usados para ajudar a classificar as palavras presentes no input em categorias.
### Gensim
O Gensim foi utilizado para fornecer o modelo de vetorização word2vec, componente crucial da nova versão.
### Pandas
Utilizado para abrir o arquivo CSV que contém os textos.
### Banco de dados
Foi utilizado um banco de dados públlico advindo do plataforma Kaggle, o qual continha informações sobre diversos comentários da rede social do Twitter em inglês.
## Etapas do processo de criação
### Pré-processamento
A primeira etapa do código é de pré-processar os dados textuais para dados matemáticos, o que foi feito pelo word2vec, o qual para cada sentença diferente, acabou criando um vetor único.
