# Genetic Algorithm Substitution Cypher Decryption Scheme
This study is a reproduction of the results and methodology detailed in the paper "Decrypting Substitution Ciphers with Genetic Algorithms" written by Jason Brownbridge from the Computer Science Department of the University of Cape Town which can be found [here](https://people.cs.uct.ac.za/~jkenwood/JasonBrownbridge.pdf). This is a Python program used to decrypt substitution ciphers using a training text of significant length to create a bigram table, or a table of the frequency of the occurence of adjacent characters. A plaintext file is then required, which will be encrypted by the program using a random subsitution cypher, the reason this is done by the program itself, rather than have the user provide it, is to ensure that a valid substitution cypher is used and so the validity of the end result can be evaluated by the program and reported to the user without having to evaluate the decrpyted text by hand.

Run Python3 GeneticAlgorithm.py -h for a list of commands that can be used to change the parameters as well as the training text, encrypted text and the name of the output text file.

For a more detailed breakdown of our results, and the details of genetic algorithms, check out our results here: . 

## Required Paramaters

-t or --trainText: requires a .txt file of sufficient length to train the bigram table

-i or --encryptText: requires a .txt of plain text that will be encrypted using a random substitution cipher

-o or --decryptText: requires the name of a .txt to be produced with the decrypted message

## Optional Parameters and their Default Values

* --popSize: Population Size - 500
* --tSize: Tournament Size - 20
* --pTournament: Probability of most fit individual winning tournament - .75
* --pCrossover: Probability of Crossover - .65
* --pMutation: Probability of mutations - .2
* --percentElite: Percentage Elitism - 15%
* --crossPoints: Crossover Points - 5
* -m or --maxGenerations: Maximum Number of Generations - 1000
