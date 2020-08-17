import pandas as pd #only for read .csv
import numpy as np
import math # python 3.8.3 standard package 
# the idea is inspired by github/arnab-sen
class counts:
    def __init__(self):
        # Training set
        self.trg_word_counts = {}
        self.trg_class_word_counts = {}
        self.trg_num_words_in_class = {}
        self.trg_class_occurrences = {}
        self.trg_abstract_sets = []
        self.idf_counts = {}
        # Test set
        self.Testset_frequencies = []
c = counts()

# the stop_words list is from academia.edu
stop_words = ["a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are", "area", "areas", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "backed", "backing", "backs", "be", "became", "because", "become", "becomes", "been", "before", "began", "behind", "being", "beings", "best", "better", "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "case", "cases", "certain", "certainly", "clear", "clearly" , "come", "could", "d", "did", "differ", "different", "differently", "do", "does", "done", "down", "down", "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "has", "have", "having", "he", "her", "here", "herself", "high", "high", "high", "higher", "highest", "him", "himself", "his", "how", "however", "i", "if", "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "it", "its", "itself", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "my", "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once","one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "our", "out", "over", "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing", "points", "possible", "present", "presented", "presenting", "presents", "problem", "problems", "put", "puts", "q", "quite", "r", "rather", "really", "right", "right", "room", "rooms", "s", "said", "same", "saw", "say", "says", "second", "seconds", "see", "seem", "seemed", "seeming", "seems", "sees", "several", "shall", "she", "should", "show", "showed", "showing", "shows", "side", "sides", "since", "small", "smaller", "smallest", "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still", "still", "such", "sure", "t", "take", "taken", "than", "that", "the", "their", "them", "then", "there", "therefore", "these", "they", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward", "turn", "turned", "turning", "turns", "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very", "w", "want", "wanted", "wanting", "wants", "was", "way", "ways", "we", "well", "wells", "went", "were", "what", "when", "where", "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within", "without", "work", "worked", "working", "works", "would", "x", "y", "year", "years", "yet", "you", "young", "younger", "youngest", "your", "yours", "z"]


# Text preprocessing
punc="!?.-:<>/\;,()@#$\'\""
def process(abstracts, ExtensionIDF):
    processed_abstracts = []
    for abstract in abstracts:
     #  Filter the symbols and numbers
      abstract = abstract.lower()
      abstract = abstract.rstrip()
      abstract = abstract.lstrip() 
      abstract = abstract.strip('`~ ,;][!?.:<>(*)@#$\'\"+=&^%1234567890\n')
      abstract_list = abstract.split()
      abstract_list = apply_word_filters(abstract_list)
      #abstract_list = apply_word_filters2(abstract_list)
      processed_abstracts.append(abstract_list) 
    return processed_abstracts

# Filter the stop_words, getting better.
def apply_word_filters(abstract_list):  
    abstract_list = list(filter(lambda word: word not in stop_words,abstract_list))   
    return abstract_list

# Filter the verb sences, it's getting worse 79.35% 
#def apply_word_filters2 (abstract_list):  
#    abstract_list = list(filter(lambda word: word.endswith(('ing','ed','ly'))#,abstract_list,))
#    return abstract_list

#Process the text, find a good model with  cross-validation
print("Text processing...")

# 10-folds cross-validation 
def cross_validate(k_folds = 10, ExtensionCNB = True, ExtensionIDF = True):
    print("ExtensionCNB = {}, ExtensionIDF = {}".format(ExtensionCNB, ExtensionIDF))
    # Split the training_set
    trg_ = pd.read_csv("trg.csv")
    trgs = []
    k_folds = k_folds
    accuracies = []
    for i in range(1, k_folds + 1):
        start = (i-1)*int((len(trg_)/k_folds))
        stop = i*int((len(trg_)/k_folds))
        trgs.append(trg_[start:stop])

    # Run cross-validation
    for i in range(k_folds):
        print("Fold {} of {}".format(i+1, k_folds))
        trg = pd.concat(trgs[:i] + trgs[i+1:])
        Testset_answers = trgs[i]
        trg.index = range(len(trg))
        Testset_answers.index = range(len(Testset_answers))
        Testset = Testset_answers.drop(columns = ["class"])

        #accuracy for Test_set:
        Testset_class_predictions = classify(trg, Testset, ExtensionCNB, ExtensionIDF)
        Testset["class"] = Testset_class_predictions
        correct = 0
        total = len(Testset)

        for i in range(len(Testset)):
            if list(Testset["class"])[i] == list(Testset_answers["class"])[i]:
                correct += 1
        accuracy = 100 * correct/total
        accuracies.append(accuracy)
        print()

    print("Accuracies from all {} folds with ExtensionCNB = {}, ExtensionIDF = {}".format(
        k_folds, ExtensionCNB, ExtensionIDF))
    print([round(accuracy, 2) for accuracy in accuracies])
    print("10-folds cross-validation accuracy : " + str(sum(accuracies)/len(accuracies)) + "%")
    print()

# Naive Bayes Classifier (NBC)
# it only run once on this assigment, so I use sklearn package.

# Multinomial Naive Bayes Classifier (MNBC)
def get_counts(trg, Testset):
    # To get Word's counts all in one
    for i in range(len(trg)):
        class_ = trg["class"][i]
        increment_dict(c.trg_class_occurrences, class_)
        c.trg_abstract_sets.append(set())
        for word in trg["abstract"][i]:            
            if class_ not in c.trg_class_word_counts:
                c.trg_class_word_counts[class_] = {}
            if class_ not in c.trg_num_words_in_class:
                c.trg_num_words_in_class[class_] = 0
                
            increment_dict(c.trg_word_counts, word)
            increment_dict(c.trg_class_word_counts[class_], word)
            increment_dict(c.trg_num_words_in_class, class_)
            c.idf_counts[word] = 1
            

    # To get word's frequencies from Testset
    for i in range(len(Testset)):
        c.Testset_frequencies.append(dict())
        for word in Testset["abstract"][i]:
            increment_dict(c.Testset_frequencies[i], word)

def increment_dict(dict_, key):
    if key in dict_:
        dict_[key] += 1
    else:
        dict_[key] = 1

def get_idf(word, trg):
    # ExtensionIDF (Inverse Document Frequency)
    if word in c.idf_counts:
        denominator = c.idf_counts[word]
    else:
        denominator = 1

    return math.log10(len(trg) / denominator)

def multinomial_naive_bayes(trg, Testset, ExtensionCNB, ExtensionIDF):
    print("Processing the training abstracts...")
    trg["abstract"] = process(trg["abstract"], ExtensionIDF)
    
    print("Processing the test abstracts...")
    Testset["abstract"] = process(Testset["abstract"], ExtensionIDF)

    print("Training the MNBC...")
    get_counts(trg, Testset)

    print("Classifying the test abstracts...")
    classes = set(np.unique(trg["class"]))
    class_probabilities = {}
    prior = {}
    log = math.log10
    for class_ in classes:
        prior[class_] = c.trg_class_occurrences[class_]/len(trg)
    alpha = 1
    labels = []
        
    for i in range(len(Testset)):
        class_probabilities = prior.copy()
        for word in c.Testset_frequencies[i]:   
            idf = get_idf(word, trg)
            for class_ in classes:
                frequency = c.Testset_frequencies[i][word]
                if ExtensionIDF:
                    frequency *= idf
                if word in c.trg_class_word_counts[class_]:
                    numerator = c.trg_class_word_counts[class_][word] + alpha
                else:
                    numerator = alpha
                denominator = c.trg_num_words_in_class[class_] + len(c.trg_word_counts)      
                # ExtensionCNB(Complement Naive Bayes)
                if ExtensionCNB:
                    comp_numerator = alpha
                    comp_denominator = len(c.trg_word_counts)
                    for cl in classes.difference({class_}):
                        if word in c.trg_class_word_counts[cl]:
                            comp_numerator += c.trg_class_word_counts[cl][word] + alpha
                        comp_denominator += c.trg_num_words_in_class[cl]
                    class_probabilities[class_] -= frequency * log(comp_numerator/comp_denominator)
                class_probabilities[class_] += frequency * log(numerator/denominator)
        label = max(class_probabilities, key = class_probabilities.get)
        labels.append(label)
    return labels

# Use this good model to generate classifications
def classify(trg, Testset, ExtensionIDF = True, ExtensionCNB = True):
    c.__init__()
    return multinomial_naive_bayes(trg, Testset, ExtensionCNB, ExtensionIDF)

    
# Write the test set classsifications to a .csv so it can be submitted to Kaggle
def write_output(ExtensionCNB = True, ExtensionIDF = True):
    trg = pd.read_csv("trg.csv")
    Testset = pd.read_csv("tst.csv")
    Testset["class"] = classify(trg, Testset, ExtensionCNB, ExtensionIDF)
    Testset.drop(["abstract"], axis = 1).to_csv("tst_kaggle.csv", index = False)
    Testset.head()


if __name__ == "__main__":
    folds = 10
   # cross_validate(k_folds = folds, ExtensionCNB = False, ExtensionIDF = False)
   # cross_validate(k_folds = folds, ExtensionCNB = True, ExtensionIDF = False)
   # cross_validate(k_folds = folds, ExtensionCNB = False, ExtensionIDF = True)
    cross_validate(k_folds = folds, ExtensionCNB = True, ExtensionIDF = True)
    
    write_output(ExtensionCNB = True, ExtensionIDF = True)