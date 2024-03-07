import glob
import os
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def load_data(directory):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory,"HAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(0)
    for f in glob.glob(os.path.join(directory,"SPAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(1)
    
    
    return x,y

#normilizing the dataset
def normilize(x , y , num_of_deleted_ham = 0):
    count = 0
    i = 0

    while count < num_of_deleted_ham :
        if y[i] == 0:
            x.remove(x[i])
            y.remove(y[i])
            count+=1
        else:
             i+=1
    
    return x , y

    


def tokenization(doc):
     # Tokenize the text
    tokens = word_tokenize(doc)

    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens ]

    return [token.lower() for token in stemmed_tokens if not token in string.punctuation and not token.lower() in stop_words]

#count words frequency in spam and ham mails
def freq( x , y):
    spam_fd ={}
    ham_fd ={}
    spam_count = 0
    ham_count = 0
    for i in range(len(x)):
            #count spam/ham emails
            if y[i] == 0:
                  ham_count = ham_count + 1
            else:
                  spam_count = spam_count + 1
            
            #Binarization
            temp_dic = {}
            #tokenization
            words = tokenization(x[i])
            for word in words:
                  
                  if word in temp_dic:
                        continue
                  
                  temp_dic[word] = 1 
                  if y[i] == 0:
                        if word in ham_fd:
                              ham_fd[word] = ham_fd[word] + 1 
                        else:
                              ham_fd[word] = 1
                  else:
                        if word in spam_fd:
                              spam_fd[word] = spam_fd[word] + 1 
                        else:
                              spam_fd[word] = 1
    
    return spam_fd , ham_fd , spam_count , ham_count

def nb_train(x, y):
    spam_fd , ham_fd , spam_count , ham_count = freq(x , y)

    return {'ham_count' : ham_count , 'spam_count' : spam_count , 'ham_fd' : ham_fd , 'spam_fd' : spam_fd}


def prob(doc , num_of_words, class_name ,  num_of_class , num_of_all_classes , smoothing , num_of_all_words , use_log):
    
    #tokenization
    words = tokenization(doc)

    if use_log:
        prob =0
        if smoothing:

            for word in words:

                if word in class_name:
                    prob += math.log(((class_name[word] + 1) / (num_of_words + num_of_all_words)))
                else:
                    prob += math.log(((0 + 1) / (num_of_words + num_of_all_words)))

        else:
            for word in words:
                
                if word in class_name:
                    prob += math.log(((class_name[word]) / (num_of_words)))
                    
                else:
                    return float('-inf')   
                        
        return math.log((num_of_class / num_of_all_classes)) + prob
    
    else:
        prob = 1.0
        if smoothing:
            
                    
            for word in words:
                
                if word in class_name:
                    prob *= ((class_name[word] + 1) / (num_of_words + num_of_all_words))

                else:
                    prob *= ((0 + 1) / (num_of_words + num_of_all_words))
        else:

            for word in words:
                
                if word in class_name:
                    prob *= (class_name[word] / num_of_words)
                else:
                    return 0
        
        return (num_of_class / num_of_all_classes) * prob
    
def nb_test(docs, trained_model, use_log = False, smoothing = False):
    y_predicted = []
    num_of_emails = trained_model['ham_count'] + trained_model['spam_count']
    num_of_words_spam = sum(model['spam_fd'].values())
    num_of_words_ham = sum(model['ham_fd'].values())

    for doc in docs:
        spam_prob = prob(doc , num_of_words_spam , trained_model['spam_fd'] , trained_model['spam_count'] , num_of_emails , smoothing , num_of_words_spam + num_of_words_ham , use_log)
        ham_prob =  prob(doc , num_of_words_ham , trained_model['ham_fd'] , trained_model['ham_count'] , num_of_emails , smoothing , num_of_words_spam + num_of_words_ham , use_log)
        y_predicted.append(0 if ham_prob >= spam_prob else 1)
    
    return y_predicted

def f_score(y_true, y_pred):
    TP = 0
    FP = 0
    FN = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        elif y_pred[i] == 1 and y_true[i] == 1:
            TP += 1

    if TP + FP == 0 or TP + FN == 0:
        return 0

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return P , R , 2 * P * R / (P + R) if (P + R) > 0 else 0





x_train, y_train = load_data('D:/sz0d/main/Desktop/programming/python/projects/NLP/SPAM/project2 (1)/project2/SPAM_training_set')
x_train ,y_train = normilize(x = x_train , y = y_train , num_of_deleted_ham=7800)
model = nb_train(x_train,y_train)

x_test, y_test = load_data("D:/sz0d/main/Desktop/programming/python/projects/NLP/SPAM/project2 (1)/project2/SPAM_test_set")
y_pred = nb_test(x_test, model , smoothing=False , use_log=False)
P , R , f_1 = f_score(y_test,y_pred)
print(f'(smoothing: {False} - use_log: {False}):')
print(f'Precision: {P} - Recall: {R}')
print(f'f-1 score: {f_1}')

print()

y_pred = nb_test(x_test, model , smoothing=True , use_log=False)
P , R , f_1 = f_score(y_test,y_pred)
print(f'(smoothing: {True} - use_log: {False}):')
print(f'Precision: {P} - Recall: {R}')
print(f'f-1 score: {f_1}')

print()

y_pred = nb_test(x_test, model , smoothing=False , use_log=True)
P , R , f_1 = f_score(y_test,y_pred)
print(f'(smoothing: {False} - use_log: {True}):')
print(f'Precision: {P} - Recall: {R}')
print(f'f-1 score: {f_1}')

print()

y_pred = nb_test(x_test, model , smoothing=True , use_log=True)
P , R , f_1 = f_score(y_test,y_pred)
print(f'(smoothing: {True} - use_log: {True}):')
print(f'Precision: {P} - Recall: {R}')
print(f'f-1 score: {f_1}')