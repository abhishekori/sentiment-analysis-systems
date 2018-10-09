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
    
runDiagnostics(getReviewSenti(vaderSentiment))
    
    




    