from nltk.sentiment import vader


sia = vader.SentimentIntensityAnalyzer()
sia.polarity_scores("this place is disgusting")
posFilePath='./rt-polaritydata/rt-polaritydata/rt-polarity.pos'
negFilePath='./rt-polaritydata/rt-polaritydata/rt-polarity.neg'
#print(sia['compoud'])/home/abhishek/ML/sentiment analysis/rt-polaritydata/rt-polaritydata

with open(posFilePath, encoding="latin-1") as f:
    posData=f.readlines()
    
with open(negFilePath,encoding="latin-1") as f:
    negData=f.readlines()
    
    
    
    
def vaderSentiment(review):
    return sia.polarity_scores(review)['compound']

def getReviewSenti(sentimentCalc):
    negReviews=[sentimentCalc(negReview) for negReview in negData]
    posReviews=[sentimentCalc(posReview) for posReview in posData]
    
    return {'res-on-pos':posReviews,'res-on-neg':negReviews}

def runDiagnostics(reviewResults):
    posReviews=reviewResults['res-on-pos']
    negReviwes=reviewResults['res-on-neg']
    corctPosReviw=float(sum(x>0 for x in posReviews))
    corctNegReviw=float(sum(x<0 for x in negReviwes))
    perctTruePos=corctPosReviw/len(posReviews)
    perctTrueNeg=corctNegReviw/len(negReviwes)
    totalAccurate = corctPosReviw + corctNegReviw
    total = len(posReviews) + len(negReviwes)
    print("Accuracy of positive reviews="+"%.2f"%(perctTruePos*100)+"%")
    print("Accuracy of negetive reviews="+"%.2f"%(perctTrueNeg*100)+"%")
    print("Overall accuracy="+"%.2f"%(totalAccurate*100/total)+"%")
    
#runDiagnostics(getReviewSenti(vaderSentiment))
from nltk.corpus import sentiwordnet as swn
def superNaiveSentiment(review):
    
    rPol=0.0
    numExp=0
    for word in review.lower().split():
#        print("word "+word)
        weight=0.0
        try:
            common_mean = list(swn.senti_synsets(word))[0]
            #print("cm "+common_mean)
            if common_mean.pos_score()>common_mean.neg_score():
                weight+=common_mean.pos_score()
            elif common_mean.pos_score()<common_mean.neg_score():
                weight-= common_mean.neg_score()
        except:
#            print('ex')
            numExp+=1
        rPol+=weight
            
    return rPol

from nltk.corpus import stopwords
from string import punctuation
stopWords = list(list(stopwords.words('english'))+list(punctuation))
def naiveSentiment(review):
    rPol=0.0
    numExp=0
    for word in review.lower().split():
#        print("word "+word)
        
        numMeanings=0
        if word in stopWords:
            continue
        weight=0.0
        try:
            for meaning in swn.senti_synsets(word):
#                
#            common_mean = list(swn.senti_synsets(word))[0]
#            #print("cm "+common_mean)
                if meaning.pos_score()>meaning.neg_score():
                    weight+=(meaning.pos_score()-meaning.neg_score())
                    numMeanings+=1
                elif meaning.pos_score()<meaning.neg_score():
                    
                    weight-= (meaning.neg_score()-meaning.pos_score())
                    numMeanings+=1
        except:
#            print('ex')
            numExp+=1
        if numMeanings>0:
            rPol = rPol +(weight/numMeanings)
            
#        rPol+=weight
            
    return rPol

    
    

#superNaiveSentiment('you are ugly but brilliant awesome fun')            
runDiagnostics(getReviewSenti(naiveSentiment))
    




    