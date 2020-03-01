import os, random, time, string
from wordcloud import STOPWORDS
from nltk.stem import PorterStemmer
from decimal import Decimal

class SpamClassifier:

    def __init__(self):
        self.__stemmer = PorterStemmer()

    def train(self, spamTrainingSet, hamTrainingSet):
        self.__lengthOfSpamTrainingSet = len(spamTrainingSet)
        self.__lengthOfHamTrainingSet = len(hamTrainingSet)
        processedSpamTrainingSet = self.__preprocessMessages(spamTrainingSet)
        processedHamTrainingSet = self.__preprocessMessages(hamTrainingSet)
        self.__probablities = self.__calculateExtimations(processedSpamTrainingSet, processedHamTrainingSet)

    def __preprocessMessages(self, messages):
        allMessagesSet = []
        
        for message in messages:
            # clear email subject and convert to words dict.
            clean_words = self.__clear(message)
            # append ony if contains words.
            if not bool(clean_words): 
                continue
            allMessagesSet.append(clean_words)
        return allMessagesSet

    # Clear text (remove non-alpha characters and conver to lowercase)
    def __clear(self, message):
        words = dict()
        STOPWORDS.add('subject')
        STOPWORDS.add('from')
        STOPWORDS.add('to')
        STOPWORDS.add('cc')
        STOPWORDS.add('bcc')
        for word in message.split():
            stemmed = self.__stemmer.stem(word.lower())
            if (stemmed not in STOPWORDS 
            and stemmed.isalpha() 
            and stemmed not in string.punctuation):
                words[stemmed] = words.get(stemmed, 0) + 1
        return words

    # Create estimation lookup for each word in training data
    def __calculateExtimations(self, spamDataSet, hamDataSet):
        spamWordDict = dict()
        hamWordDict = dict()
        wordDict = dict(dict())

        for words in spamDataSet:
            for word in words:     
                spamWordDict[word] = spamWordDict.get(word, 0) + 1        

        for words in hamDataSet:
            for word in words:     
                hamWordDict[word] = hamWordDict.get(word, 0) + 1   
    
        allWords = set(hamWordDict).union(set(spamWordDict))
        for word in allWords:
            spamVal = spamWordDict.get(word, 0)
            hamVal = hamWordDict.get(word, 0)
            wordDict[word] = {
                'spam': ((spamVal + 1) / (len(spamDataSet) + 2)),
                'ham': ((hamVal + 1) / (len(hamDataSet) + 2))  
                }
        return wordDict

    # Calculate posterior probablity
    def __calculateProbablityOfSpam(self, probablities, mail):
        # clear mail
        words = self.__clear(mail)
        
        # Pr({...}|spam)
        prOfItemsAsSpam = Decimal(1)

        # Pr({...}|ham)
        prOfItemsAsHam = Decimal(1)

        for word, value in probablities.items():
            sValue = Decimal(value['spam'])
            hValue = Decimal(value['ham'])
            prOfItemsAsSpam *= (sValue if word in words else 1 - sValue)
            prOfItemsAsHam *= (hValue if word in words else 1 - hValue)

        sCount = self.__lengthOfSpamTrainingSet
        hCount = self.__lengthOfHamTrainingSet

        # Pr(spam)
        prSpam = Decimal((sCount + 1) / (sCount + hCount + 2))
        
        # Pr(ham)
        prHam = Decimal((hCount + 1) / (sCount + hCount + 2))

        result = Decimal(0)
        result = (prOfItemsAsSpam * prSpam) / (prOfItemsAsSpam * prSpam + prOfItemsAsHam * prHam)

        return result
    
    def test(self, spamTestSet, hamTestSet):
        TP = 0
        TN = 0 
        FN = 0
        FP = 0

        for mail in spamTestSet:
            sVal = self.__calculateProbablityOfSpam(self.__probablities, mail)
            if sVal > 0.5: TP += 1 
            elif sVal < 0.5: FN += 1

        for mail in hamTestSet:
            hVal = self.__calculateProbablityOfSpam(self.__probablities, mail)
            if hVal < 0.5: TN += 1 
            elif hVal > 0.5: FP += 1

        print('\nConfusion Matrix Values:')
        print("TP: %s" % str(TP))
        print("FP: %s" % str(FP)) 
        print("TN: %s" % str(TN))         
        print("FN: %s" % str(FN)) 
        accuracy = int(( (TP + TN) / (TP + FP + TN + FN)) * 100)
        print('Accuracy: %s%%' % str(accuracy))

# Read data on related folder
def load_data_in(folder_name):
    spamList = []
    hamList = []
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(root_dir,folder_name)
    for dirpath, _, filenames in os.walk(root_path):
        folderName = os.path.split(dirpath)[1]
        for filename in filenames:
            if not filename.endswith('txt'): continue
            with open(os.path.join(dirpath, filename), encoding='latin-1') as f:
                messageBody = f.read().replace('\n','')                
                if folderName == 'ham':
                    hamList.append(messageBody)
                elif folderName == 'spam':
                    spamList.append(messageBody)          
    return (spamList, hamList)

def split(list):
    # Split dataset 70% and 30% for training/testing
    random.shuffle(list)
    splitIndex = int(len(list) * .7)
    training = list[:splitIndex]
    test = list[splitIndex:]
    return (training, test)

if __name__ == "__main__":    
    datasets = ['enron1', 'enron6']
    
    for dataset in datasets:        
        start_time = time.time()        
        print('\nLoading %s dataset...⏳' % dataset)
        spamList, hamList = load_data_in(dataset)   
        print('%s files read completed ✅' % dataset)

        ham_training, ham_test = split(hamList)
        spam_training, spam_test = split(spamList)
        print('Ham (train:test) -> %d:%d' % (len(ham_training), len(ham_test) ))
        print('Spam (train:test) -> %d:%d' % (len(spam_training), len(spam_test) ))

        classifier = SpamClassifier()

        print('Training %s dataset..⏳' % dataset)
        classifier.train(spam_training, ham_training)
        print('Training completed ✅')

        print('Testing %s dataset..⏳' % dataset)
        classifier.test(spam_test, ham_test)
        print('\nTesting completed ✅')

        print("Elapsed time: %s seconds.\n" % int(time.time() - start_time))
