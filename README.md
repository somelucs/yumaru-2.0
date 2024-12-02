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
A primeira etapa do código é de pré-processar os dados textuais para dados matemáticos, o que foi feito pelo word2vec, o qual para cada palavra, cria um valor numérico único.
### Processo de categorização e execução
#### Categorização
As palavras em um texto podem pertencer a um ou mais assuntos diferentes, e quais e quão distantes são esses assuntos pode determinar o quanto provável é de duas ou mais palavras serem similares ou não. Considerando isso, para criar um efeito parecido com esse, foi usado primeiramente 4 arrays para representarem os assuntos e a frequência deles no input, como mostra o código a seguir:
```
def cat(inp):
    #Definindo pesos dos assuntos
    Pnews=1
    Phum=1
    Phob=1
    Prev=1

    #Definindo tensor do input
    tns=torch.tensor(modelo.wv[inp.split(' ')])

    #Categorizando tensor
    #Definindo pesos finais
    for i in range(len(inp)):
        if inp[i] in news:
            Pnews+=1
    for i in range(len(inp)):
        if inp[i] in hum:
            Phum+=1
    for i in range(len(inp)):
       if inp[i] in hob:
            Phob+=1
    for i in range(len(inp)):
        if inp[i] in rev:
            Prev+=1

    #Aplicando pesos ao modelo
    #Redimensionando o tensor
    if len(modelo.wv[inp.split(' ')])%(Pnews*Phob*Phum*Prev)==0:
        tnsR=tns.view(Pnews,Phum,Phob,Prev,-1)

        #Criando um novo tensor a partir do redimensionado
        tnsN=torch.tensor(tnsR.mean(dim=0).numpy()[0]+tnsR.mean(dim=1).numpy()[0]+tnsR.mean(dim=2).numpy()[0]+tnsR.mean(dim=3).numpy()[0]+tnsR.mean(dim=4).numpy()[0])

        return tnsN/5
    else:
        return 'N'

```
As variáveis 'news','hum','hob','rev' representam arrays contendo as palavras referentes a assuntos específicos, como a 'news', que se refere às palavras usadas em notícias. Além disso, vale destacar o uso de tensores, que tem como função aqui e no geral de ser um formato usado na transformação dos valores criados pelo word2vec, uma vez que oferece uma variedade de rearranjo de dados, como se observa na parte final do código, em que o tensor referente ao input é transformado de maneira a ter suas dimensões alteradas pelos valores de frequência de assuntos. Porém, ainda é possível que a palavra não se enquadre em nenhum dos assuntos criados, o que leva a criação de uma quinta dimensão no tensor, que é ajustada automaticamente nesse caso. Por fim, as médias em cada dimensão são somadas em colocadas em um único tensor.
#### Execução
Finalizando, há ainda o processo de executar e lidar com o resultado final com base na categorização e no input. Seguindo, o processo para lidar com os dados estabelecidos foi semelhante ao usado na categorização, os quais em ambos são usados a redimensionalização de tensores.<br><br>
Para classificar se um tensor é parecido com outro, e logo o valor da palavra também, foi utilizado o argumento mínimo dele, nesse caso os tensores usados foram o tensor do vocabulário tokeinizado antes e o valor da média entre o tensor criado na categorização e o do input, no intuito de explorar e comparar a diferença entre eles para achar a maior similaridade possível.
## Conclusões
Os outputs acabam gerando palavras normalmente usadas em conjunto por um humano, por exemplo, para um input='im feeling joy in my room' o sistema retorna um output='that but had over convinced hate drop'. Esse modelo devido não utilizar técnicas mais sofisticadas pode ser superado por outros de linguagem natural, mas ainda sim pode ser útil para explorar as condições do pytorch e Python no geral.



