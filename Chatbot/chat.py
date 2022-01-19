import nltk
# nltk ----- natural language toolkit
import numpy as np
import string
import random

f=open("chatbot.txt")
raw= f.read()
#print(raw)

# text cleaning
# lower case
raw=raw.lower()
sent_tokens= nltk.sent_tokenize(raw)

word_tokens=nltk.word_tokenize(raw)

#print(word_tokens)
#print(sent_tokens)

#print(sent_tokens[:2])

# stemming or lemnmation
lemmer =  nltk.stem.WordNetLemmatizer()

#  lemmationzation ---- convert the word into dictionary or base format

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct),None) for punct in string.punctuation)
# $ ! @ & .....
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))
GREETING_INPUTS= ("hello","hi","what's up","hey")
GREETING_RESPONSES = ["hi","hello","Iam glad! you are taking to me","how may i will help you"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response=""
    sent_tokens.append(user_response)
    TfidfVec= TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx= vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if req_tfidf==0:
        chatbot_response=chatbot_response+"Iam sorry unable to understand question"
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]

    return chatbot_response

flag=True

while(flag):
    user_response=input()
    user_response=user_response.lower()
    if user_response!='bye':
        if user_response=='thanks' or user_response=="thank you":
            flag=False
            print("System : You are welcome")
        else:
            if greeting(user_response)!=None:
                print("System : "+ greeting(user_response))
            else:
                print("System : ",end=" ")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("System : Bye ! take care ")

