#-------------------------------------------------------------------------
# AUTHOR: Janista Gitbumrungsin
# FILENAME: search_engine
# SPECIFICATION: Mimic the process of a search engine finding index terms and calculating the document scores to rank based on a query
# FOR: CS 4250- Assignment #1
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard arrays

#importing some Python libraries
import csv
import math

documents = []
labels = []

#reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])
            labels.append(row[1])

#Conduct stopword removal.
#--> add your Python code here
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}
tokens = []

for i in range(len(documents)):
    tokens.append(documents[i].split())

for line in tokens:
    for word in line:
        if word in stopWords:
            line.remove(word)
        

#Conduct stemming.
#--> add your Python code here
stemming = {
  "cats": "cat",
  "dogs": "dog",
  "loves": "love",
}

for index, line in enumerate(tokens):
    for i, word in enumerate(line):
        if word in stemming.keys():
            line[i] = stemming.get(word)
print("\n-----SEARCH ENGINE-----\n")
print("Document Tokens:")
for line in tokens:
    print(line)
print()

#Identify the index terms.
#--> add your Python code here
terms = []

for line in tokens:
    for word in line:
        if word not in terms:
            terms.append(word)

#Build the tf-idf term weights matrix.
#--> add your Python code here
docMatrix = []
frequency = {}
idf = {}
tf = {}

for word in terms:
  frequency.update({word : 0})
  idf.update({word : 0})

for word in terms:
    for line in tokens:
        if word in line:
            frequency.update({word : frequency.get(word)+1})

for word in idf:
    div = len(documents) / frequency.get(word)
    value = math.log(div, 10)
    idf.update({word : value})

for i in range(len(documents)):
    tf.update({i : {}})

for index, line in enumerate(tokens):
    for word in terms:
        count = line.count(word)
        total = len(line)
        result = count/total
        tf.get(index).update({word : result})

for entry in tf.values():
    for word in terms:
        result = entry.get(word) * idf.get(word)
        docMatrix.append(result)  

count = 1
print("-----tf-idf Matrix-----")
print("\t", end = "")
for word in terms:
    print(word, end = "   ")
print()

for i in range(len(docMatrix)):
    if i != 0 and i%len(terms) == 0:
        print("\n")  
    if i%len(terms) == 0:
        print("Doc " + str(count) + ":", end = " ")
        count = count + 1
    print("%.3f" % (docMatrix[i]), end = " ")
print("\n")
    

#Calculate the document scores (ranking) using document weigths (tf-idf) calculated before and query weights (binary - have or not the term).
#--> add your Python code here
docScores = []
queryWeights = []
query = "cats and dogs"

queryTokens = query.split()
for i, word in enumerate(queryTokens):
        if word in stopWords:
            queryTokens.remove(word)
for i, word in enumerate(queryTokens):
        if word in stemming.keys():
            queryTokens[i] = stemming.get(word)

print("Query Tokens:", end = " ")
print(queryTokens)

for word in terms:
    if word in queryTokens:
        queryWeights.append(1)
    else:
        queryWeights.append(0)

print("Query Weights:", end = " ")
print(queryWeights)
print()

sum = 0
for i in range(len(docMatrix)):
    if (i%len(terms) == 0):
        docScores.append(sum)
        sum = 0
    value = docMatrix[i] * queryWeights[i%len(terms)]
    sum = sum + value
docScores.append(sum)
docScores.pop(0)

print("Document Scores:")
print(docScores)
print()

#Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
#--> add your Python code here
for i in range(len(labels)):
    labels[i] = labels[i].strip()

hits = 0
misses = 0
noise = 0

print("Retrieved Documents:")
for i in range(len(docScores)):
    if docScores[i] >= 0.1:
        print("Document " + str(i + 1))
print()

for i in range(len(labels)):
    if docScores[i] >= 0.1 and labels[i] == "R":
        hits = hits + 1
    elif docScores[i] >= 0.1 and labels[i] == "I":
        noise = noise + 1
    elif docScores[i] < 0.1 and labels[i] == "R":
        misses = misses + 1

recall = (hits /(hits + misses)) * 100
precision = (hits / (hits + noise)) * 100
print("Precision: " + str(precision) + " %")
print("Recall: " + str(recall) + " %")
print("---------------")