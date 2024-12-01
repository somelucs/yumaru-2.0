import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from nltk.corpus import brown
import math


app = Flask(__name__)
nltk.download('punkt_tab')
nltk.download('brown')
snt = pd.read_csv('D:/pyP2/emt.csv')['Comment'][:300]
#Palavras com tom de notícia
news=brown.words(categories='news') 
#Palvras com tom de 'hobbie'
hob=brown.words(categories='hobbies')
#Palvras com tom de humor
hum=brown.words(categories='humor')
#Palvras com tom de review
rev=brown.words(categories='reviews')



sntN=[]
for i in range(len(snt)):
    sntN.append(snt[i].split(' '))


# Pré-processamento de texto
def preproces(texto):
    
    tokens = word_tokenize(texto)
    tokens = [w for w in tokens if w.isalpha()]
    return tokens

#Sentenças pré-processadas
preproces_s = [preproces(sentença) for sentença in snt]
modelo = Word2Vec(sentences=preproces_s, vector_size=100, window=5, min_count=1, workers=4, sg=1)
word_vectors = modelo.wv[(modelo.wv.index_to_key)]  

#Categorizar input
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
    
#Definindo função de execução do modelo
def treino(inp):
    mtx=[]
    #String do output
    strOut=''
    rgs=''
    for i in range(1,len(inp.split(' '))-1):
        #Definindo tensores
        if len(modelo.wv[inp.split(' ')])%i==0 and len(word_vectors)%i==0:#Vendo se 'i' se enquadra como dimensão do tensor
            tns=torch.tensor([word_vectors]).view(i,-1)#Tensor do vocabulário
            tns2=torch.tensor(modelo.wv[inp.split(' ')]).view(i,-1)#Tensor do input

            #Comparando tns2 com o tensor criado por cat()
            tnsCat=cat(inp)
            tns2Md=tns2.mean(dim=1).min()#Menor valor do tensor para fins comparativos/criação do tensor médio
            if tnsCat!='N':
                if tns2.mean(dim=1).min()>=tnsCat.mean(dim=0).min():
                    tns2Md+=tnsCat.mean(dim=0).min()
                tns2Md/=2#Realizando a operação de médi
            
            f=[]#Matriz de index

            #Comparando os tensores de vocabulário com o tensor médio do input
            for j in range(len(word_vectors)-1):
                if torch.tensor(word_vectors[j]).min()>=tns2Md.min():
                    f.append(j)
                
            
            #Comparando e achando os vetores mais próximos do input
            for w in range(1,(len(f)-1)):
                for z in range(f[w-1],f[len(f)-1]):
                    if len(word_vectors)%z==0 and z<len(word_vectors) and z==f[w]:
                        strOut+= str(list(modelo.wv.index_to_key)[w])+' '#Mostrando palavra formada
    return strOut

#Input
sttr=input('Sua frase:')
print(treino(sttr.lower()))

@app.route('/')
def index():
    return render_template('menu.html')

if __name__ == '__main__':
    app.run(debug=True)
