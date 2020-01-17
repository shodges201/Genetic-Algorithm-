import sys
import math
import random
import getopt

def preprocess(train):
    line = train.readline()
    newTrain = ""
    while line:
        line = line.strip(" ")
        line = line.replace("\n", " ")
        if(line == " "):
            line = train.readline()
            continue
        i = 0
        newLine = ""
        # remove everything besides letters and spaces (punctuation, quotations, etc.)
        for char in line:
            if(char == ' '):
                newLine = newLine + char
            if((ord(char) > 64 and ord(char) < 91)):
                newLine = newLine + char
            if((ord(char) > 96 and ord(char) < 123)):
               newLine = newLine + char
               
        line = newLine
                
        newTrain = newTrain + line
        line = train.readline()
        
    return newTrain
            

def biGrams(biDict, train):
    i = 0
    j = 2
    while(j < len(train)):
        digram = train[i:j]
        if(" " in digram):
            i += 1
            j += 1
            continue
        else:
            key = digram[0].upper() + digram[1].upper()
            biDict[key] = biDict[key] + 1
        
            i += 1
            j += 1
            
    return biDict

def frequencyTable(freqDict):
    for key in freqDict.keys():
        val = freqDict[key]
        if val != 0:
            freqDict[key] = math.log(val, 2.0)
        
    return freqDict

def fitness(biDict, freqDict):
    biSum = 0
    for key in biDict.keys():
        if(biDict[key] != 0):
            biSum += biDict[key] * freqDict[key]
            
    return biSum

def encrypt(alphabet, mixedAlph, text):
    encryptedText = ""
    for item in text:
        if(item.upper() != " "):
            idx = alphabet.index(item.upper())
            encryptedText += mixedAlph[idx]
        else:
            encryptedText += " "
            
    return encryptedText

def generatePop(pop, popSize, alphabet):
    for i in range(popSize):
        temp = alphabet.copy()
        random.shuffle(temp)
        pop.append(temp)
    
    return pop

def crossover(p1, p2, crossPoints):
    crossPointList = list(range(0,26))
    
    points = random.sample(crossPointList, crossPoints)
    points.sort()
    child1 = [""] * 26
    child2 = [""] * 26
    p1Copy = p1.copy()
    p2Copy = p2.copy()

    for i in range(crossPoints):
        child1[points[i]] = p1[points[i]]
        child2[points[i]] = p2[points[i]]
        
        p1Copy.remove(p2[points[i]])
        p2Copy.remove(p1[points[i]])

    for i in range(len(child1)):
        item1 = child1[i]
        item2 = child2[i]
        if(item1 == ''):
            child1[i] = p2Copy[0]
            p2Copy.pop(0)
            child2[i] = p1Copy[0]
            p1Copy.pop(0)
    
    return child1, child2

def mutation(child): #Takes a child after crossover
    spots = random.sample(range(0,26), 2)   #Spots to swap
    temp = child[spots[0]]
    child[spots[0]] = child[spots[1]]
    child[spots[1]] = temp
    return child

def Tournament(participants, popFits, p, size):
    winner = 0
    pLimit = p
    place = 0
    probability = float("inf")
    maxScore = 0
    winnerLocation = 0
    while(probability >= pLimit and place < len(participants)):
        place += 1
        probability = random.random()
        if(probability > pLimit):
            pLimit = pLimit * (1-p)
    if(place == 1):
        maxScore = max(popFits)
        winnerLocation = popFits.index(maxScore)
        winner = participants[winnerLocation]

    else:
        for i in range(0, place-1):
            maxScore = max(popFits)
            winnerLocation = popFits.index(maxScore)
            popFits.pop(winnerLocation)
            participants.pop(winnerLocation)
        maxScore = max(popFits)
        winnerLocation = popFits.index(maxScore)
        winner = participants[winnerLocation]
        
    return winner

def main():
    popSize = 500
    size = 20
    pTourn = .75
    pCross = .65
    pMut = .2
    percentElite = 15
    crossPoints = 5
    maxGenerations = 1000
    err = "The valid arguments are:\n-t or --trainText for the name of the training text\n-i or --encryptText for the name of the encrpyted text\n-o or --decryptText for the name of the output decrypted text\n--popSize for the population size\n--tSize for the tournament size\n--pTournament for the tournament probability\n--pCrossover for the crossover probability\n--pMutation for the mutation probability\n--percentElite for the percentage of elitism\n--crossPoints for the number of crossover points"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:i:o:h:m:', ['trainText=', 'encryptText=', 'decryptText=' 'popSize=', 'tSize=', 'pTournament=', 'pCrossover=', 'pMutation=', 'percentElite=', 'crossPoints=', 'maxGenerations='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print(err)
        elif opt in ('-t', '--trainText'):
            train = open(arg, "r")
        elif opt in ('-i', '--encryptText'):
            encryptText = open(arg, "r")
        elif opt in ("-o", '--decryptText'):
            decryptText = open(arg, "w")
        elif opt == '--popSize':
            popSize == int(arg)
        elif opt == '--tSize':
            size = int(arg)
        elif opt == '--pTournament':
            pT = float(arg)
        elif opt == '--pCrossover':
            pC = float(arg)
        elif opt == '--pMutation':
            pM = float(arg)
        elif opt == '--percentElite':
            percentElite = int(arg)
        elif opt == '--crossPoints':
            crossPoints = int(arg)
        elif opt in ('-m', '--maxGenerations'):
            maxGenerations = arg
            maxGenerations = int(maxGenerations)
        else:
            print(err)
            sys.exit(2)
    
    alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    freqDict = {}
    biGramDict = {}
    population = []
    
    randEncryptAlph = alph.copy()
    random.shuffle(randEncryptAlph)

    for i in range(26):
        for j in range(26):
            freqDict[alph[i] + alph[j]] = 0
            biGramDict[alph[i] + alph[j]] = 0
            
    encryptedText = open("empty.txt", "w")

    newTrain = preprocess(train)
    eText = preprocess(encryptText)

    encryptedMessage = encrypt(alph, randEncryptAlph, eText)
    encryptedText.write(encryptedMessage)
    
    freqTable = biGrams(freqDict, newTrain)
    freqTable = frequencyTable(freqTable)

    population = generatePop(population, popSize, alph)

    fitScores = []
    generation = 0
    tournamentPop = [] # participants in the tournament
    tournamentFit = [] # fitnesses of participants
    maxFitness = 0
    convergeCount = 0
    numberElite = len(population) // percentElite
    bestKey = []
    while(generation < maxGenerations): # sets a maximum number of generations
        fitScores = []
        nextPop = []
        #creates the table of fitness scores for each individual
        for key in population:
            fitnessScore = 0
            decryptedMessage = ""
            biTable = biGramDict.copy()
            decryptedMessage = encrypt(alph, key, encryptedMessage)
            biTable = biGrams(biTable, decryptedMessage)
            fitnessScore = fitness(biTable, freqTable)
            fitScores.append(fitnessScore)
        parents = [] # list of two winners of tournament
        
        if(max(fitScores) > maxFitness): # checking for max fitness score
            maxFitness = max(fitScores)
            idxBest = fitScores.index(max(fitScores))
            bestKey = population[idxBest]
        elif(maxFitness == max(fitScores)):
            convergeCount += 1
        '''    
        if(convergeCount > 20): # checking for convergence
            break
        '''
        for i in range(0, numberElite - 1): # adds the top elite to the next generation
            topFit = 0
            topIdx = 0
            topFit = max(fitScores)
            topIdx = fitScores.index(topFit)
            nextPop.append(population[topIdx])
            population.pop(topIdx)
            fitScores.pop(topIdx)
        while(len(population) != 0 and len(population) != 1): #tournament process
            parents = []
            i = 0
            while i < 2 and len(population) != 0:
                tournamentPop = []
                tournamentFit = []
                if(len(population) < size):
                    tournamentPop = random.sample(range(0,len(population)), len(population)) #fills population of tournament to size of tournament
                else:
                    tournamentPop = random.sample(range(0,len(population)), size)
                for item in tournamentPop:
                    tournamentFit.append(fitScores[item])
                parent = Tournament(tournamentPop, tournamentFit, pTourn, size) # returns the index of the winner from the tournament (parent)
                parents.append(population[parent]) #adds the parents key to the list
                population.pop(parent) #removes the parent from the population
                fitScores.pop(parent) #removes fitScore of parent from list
                i += 1
            chanceCross = random.random()
            if(chanceCross < pCross):
                child1, child2 = crossover(parents[0], parents[1], crossPoints) #performs crossover on the parents
            else:
                child1 = parents[0]
                child2 = parents[1]
            parents = []
            parents.append(child1)
            parents.append(child2)
            for i in range(2):
                chanceMut = random.random()
                if(chanceMut < pMut):
                    mutation(parents[i])
            nextPop.append(child1)
            nextPop.append(child2)
        
        population = nextPop.copy()
        generation += 1

    print("number of generations: " + str(generation))
    print("maxFitness Score: " + str(maxFitness))
    count = 0
    for i in range(26):
        if(bestKey[i] == randEncryptAlph[i]):
            count += 1
            
    decryptedMessage = encrypt(alph, bestKey, encryptedMessage)
    print("percent of key correct: " + str(count / 26) + " %")
    decryptText.write(decryptedMessage)
    
main()
