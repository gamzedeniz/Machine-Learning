import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

import math
df = pd.read_csv('all_sentiment_shuffled.txt', delimiter="\t", header=None,names=["Topic", "Label", "Doc", "Comment"])
#taking the data from the text file and diving it into the columns
df["Label"]=(df["Topic"].str.split(" ", n=3,expand = True))[1]
df["Doc"]=(df["Topic"].str.split(" ", n=3,expand = True))[2]
df["Comment"]=(df["Topic"].str.split(" ", n=3,expand = True))[3]
df["Topic"]=(df["Topic"].str.split(" ", n=3,expand = True))[0]
train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=False) #splitting 80% of data as train 20% of data as test


def bagofwordsuni():
    c_vecuni = CountVectorizer(ngram_range=(1,1))
    return c_vecuni

def bagofwordsbi():
    c_vecbi= CountVectorizer(ngram_range=(2,2))
    return c_vecbi

def bigramstopwords():
    c_vecbis= CountVectorizer(stop_words='english',ngram_range=(2,2))
    return c_vecbis

def unigramstopwords():
    c_vecunis= CountVectorizer(stop_words='english',ngram_range=(1,1))
    return c_vecunis

def tfidf():
    vec= TfidfVectorizer()
    return vec

def tfidfstop():
    vec= TfidfVectorizer(use_idf=True,stop_words='english')
    return vec


def ByTag(column,data,tag): #creating a list for the associated label
    list=[]
    for index,row in data.iterrows():
        if row[column] == tag:
            list.append(row["Comment"])
    return list

def NaiveBayes(freq,comments,totallabel,totalwords,check): #takes each test comment and frequencies of each word for that class
    prob=0 #tootallabel as the number of words in that label
    if(check==1):
        comment=comments.split()#for unigram
    else:
        comment=[i[0]+" "+i[1] for i in zip(comments.split(" ")[:-1], comments.split(" ")[1:])]#for bigram
    for word in comment: #calculates likehood
        if word in freq.keys():
            count = freq[word]
        else:
            count = 0
        probs=(count + 1)/(totallabel+ totalwords) #laplace smoothing
        #getting log probability
        prob+=math.log(probs) #adding them up by log multplication rules
    return prob

def Freq(counted,list,total_features):
    X=counted.fit_transform(list)
    wordList = counted.get_feature_names()
    countList= X.toarray().sum(axis=0)
    total_features.append(countList.sum(axis=0)) #adding number of total features of that class to a list to store it
    freqs = dict(zip(wordList,countList))

    probs=[]
    for word,count in zip(wordList,countList):
        probs.append(count/len(wordList))
    dictprob=dict(zip(wordList,probs)) #for printing out probabilities of words of that class

    i=0
    for value in sorted(freqs.items(), key=lambda kv: kv[1], reverse=True): #print top 25 according to frequencies
        i+=1
        #print(value)
        if(i==25):
            break
    #print frequencies
    #print(sorted(freqs.items(), key=lambda kv: kv[1], reverse=True))
    #for value in sorted(freqs.items(), key=lambda kv: kv[1], reverse=True):
        #print(value)
    return freqs

def Classify(pos_freq,neg_freq,check):
    totalpos=pos_total[0]
    totalneg=neg_total[0]
    count=0
    for index,row in test.iterrows():
        pos=NaiveBayes(pos_freq,row["Comment"],totalpos,total_features,check)+math.log(len(trainpos)/len(train.index))
        neg=NaiveBayes(neg_freq,row["Comment"],totalneg,total_features,check)+math.log(len(trainneg)/len(train.index))
        if(pos>neg):
            if (row["Label"]=="pos"):
                count+=1
            #else:
                #print(row["Label"],"expected pos",row["Comment"]) #to check missclassified sentences
        elif(pos<neg):
            if(row["Label"]=="neg"):
                count+=1
            #else:
                #print(row["Label"],"expected neg",row["Comment"])
    Acc=100*count/len(test.index)
    print("Accuracy of Label Classification: ", Acc)

trainpos=ByTag("Label",train,"pos") #creates class lists for storing comments
trainneg=ByTag("Label",train,"neg")

def get_total_features(): #return total features
    trainall=[row["Comment"] for index,row in train.iterrows()]
    #using associated countvector
    vec= tfidf()
    #vec=tfidfstop()
    #vec=bagofwordsbi()
    #vec=bagofwordsuni()
    #vec=bigramstopwords()
    #vec=unigramstopwords()
    vec.fit_transform(trainall)
    return len(vec.get_feature_names())


#Using the count vector I need
#vec= bagofwordsbi()
#vec= bagofwordsuni()
#vec= bigramstopwords()
#vec= unigramstopwords()
vec= tfidf()
#vec=tfidfstop()

total_features=get_total_features()#getting total features count

pos_total =[]#for storing total features count of the class
neg_total =[]

#getting freqs of all words of that class and
#geting total features count for that class
#print("Pos Freq")
pos_freqs=Freq(vec,trainpos,pos_total)

#print("Neg Freq")
neg_freqs=Freq(vec,trainneg,neg_total)

#check == 1 for unigram 0 for bigram
Classify(pos_freqs,neg_freqs,1)

#----------------------------------------------
#FOR THE BONUS PART OF THE ASSIGNMENT Classification by Topics
#------------------------------------------------
#I will be using the same count vector and total features from above

#for storing total features count of the class
ldvd=[]
lmusic=[]
lbooks=[]
lhealth=[]
lcamera=[]
lsoftware=[]

traindvd=ByTag("Topic",train,"dvd")
trainmusic=ByTag("Topic",train,"music")
trainbooks=ByTag("Topic",train,"books")
trainhealth=ByTag("Topic",train,"health")
traincamera=ByTag("Topic",train,"camera")
trainsoftware=ByTag("Topic",train,"software")

pd=Freq(vec,traindvd,ldvd)
pm=Freq(vec,trainmusic,lmusic)
pb=Freq(vec,trainbooks,lbooks)
ph=Freq(vec,trainhealth,lhealth)
pc=Freq(vec,traincamera,lcamera)
ps=Freq(vec,trainsoftware,lsoftware)

def ClassifyTopic(dvd,music,books,health,camera,software,check):
    tdvd=ldvd[0]
    tmusic=lmusic[0]
    tbooks=lbooks[0]
    thealth=lhealth[0]
    tcamera=lcamera[0]
    tsoft=lsoftware[0]
    count=0
    for index,row in test.iterrows():
        d=NaiveBayes(dvd,row["Comment"],tdvd,total_features,check)+math.log(len(traindvd)/len(train.index))
        m=NaiveBayes(music,row["Comment"],tmusic,total_features,check)+math.log(len(trainmusic)/len(train.index))
        b=NaiveBayes(books,row["Comment"],tbooks,total_features,check)+math.log(len(trainbooks)/len(train.index))
        h=NaiveBayes(health,row["Comment"],thealth,total_features,check)+math.log(len(trainhealth)/len(train.index))
        c=NaiveBayes(camera,row["Comment"],tcamera,total_features,check)+math.log(len(traincamera)/len(train.index))
        s=NaiveBayes(software,row["Comment"],tsoft,total_features,check)+math.log(len(trainsoftware)/len(train.index))
        if(d>m and d>b and d>h and d>c and d>s):
            if (row["Topic"]=="dvd"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
        elif(m>d and m>b and m>h and m>c and m>s):
            if (row["Topic"]=="music"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
        elif(b>d and b>m and b>h and b>c and b>s):
            if (row["Topic"]=="books"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
        elif(h>d and h>b and h>m and h>c and h>s):
            if (row["Topic"]=="health"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
        elif(c>d and c>b and c>h and c>m and c>s):
            if (row["Topic"]=="camera"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
        elif(s>d and s>b and s>h and s>c and s>m):
            if (row["Topic"]=="software"):
                count+=1
            #else:
                #print(row["Comment"]) #to check missclassified sentences
    Acc=100*count/len(test.index)
    print("Accuracy of Category Classification: ", Acc)

ClassifyTopic(pd,pm,pb,ph,pc,ps,1) #1 for unigram 0 for bigram
