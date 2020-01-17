# Genetic-Algorithm-
Python program used to decrypt substitution ciphers. Based off of the results from the paper "Decrypting Substitution Ciphers with Genetic Algorithms" written by Jason Brownbridge from the Computer Science Department of the University of Cape Town.

Run Python3 GeneticAlgorithm.py -h for a list of commands that can be used to change the parameters as well as the training text, encrypted text and the name of the output text file.

Necessary commands include:

-t or --trainText -> requires a .txt file of sufficient length to train the bigram table

-i or --encryptText -> requires a .txt of plain text that will be encrypted using a random substitution cipher

-o or --decryptText -> requires the name of a .txt to be produced with the decrypted message

The default parameters are:

# Population Size - 500
# Tournament Size - 20
# Probability of most fit individual winning tournament - .75
# Probability of Crossover - .65
# Probability of mutations - .2
# Percentage Elitism - 15%
# Crossover Points - 5
# Maximum Number of Generations - 1000
