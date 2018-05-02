# Nicholas Gao
# ECE467 Natural Language Processing
# Professor Sable
# Spring 2018
# Project 2: Independent POS Tagger

from nltk.corpus import brown
from nltk.corpus import treebank
import math
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
import nltk.tag

tknzr = TweetTokenizer()
smoothingParam = 0.01

class tagset:
    def __init__(self):
        self.tags = {}

    def updateTagset(self, tag, word, prevTag):
        if tag not in self.tags:
            t = self.Tag(tag)
            t.updateTagCounts(word, prevTag, True)
            self.tags[tag] = t
        else:
            # self.tags[Tag].tag_count += 1
            self.tags[tag].updateTagCounts(word, prevTag, False)

    def assignProbabilities(self):
        for t in self.tags:
            # Word Probs Assigned at Tag Level
            self.tags[t].assignWordProbabilities()
            # Transition PRobs Assigned at This Level (Tagset), Requires Knowledge of All Tags
            self.assignTransitionProbabilities(self.tags[t])

    def assignTransitionProbabilities(self, the_tag):
        trans_probs = the_tag.transition_probs
        for prev_tag_record_key in the_tag.transition_probs:
            if prev_tag_record_key not in self.tags:
                print('Error: Previous Tag not found in TagSet.')
            trans_probs[prev_tag_record_key].prob = math.log(trans_probs[prev_tag_record_key].count/self.tags[prev_tag_record_key].tag_count)

    class Tag:
        def __init__(self, key):
            self.the_key = key
            self.tag_count = 1
            self.unknownWordProbability = None # Prob for Unknown Word, Unseen TagTransition Prob will be 0.
            self.WordsSeen = 0 # Number of Total Words Seen
            self.TransitionsSeen = 0 # Number of Total Transitions Seen
            self.word_probs = {} # Number of Unique Words, Counts and Probs
            self.transition_probs = {} # Number of Unique Tag Transitions, Cs & Ps

        def updateTagCounts(self, word, prevTag, isNew):
            if isNew == False:
                self.tag_count += 1
            self.updateWordCount(word)
            if prevTag != None:
                self.updateTagTransitionCount(prevTag)
                self.TransitionsSeen += 1
            self.WordsSeen += 1

        def updateWordCount(self, word):
            if word not in self.word_probs:
                r = self.record(1)
                self.word_probs[word] = r
            else:
                self.word_probs[word].count += 1

        def updateTagTransitionCount(self, prevTag):
            if prevTag not in self.transition_probs:
                r = self.record(1)
                self.transition_probs[prevTag] = r
            else:
                self.transition_probs[prevTag].count += 1

        class record:
            def __init__(self, count):
                self.prob = None
                self.count = count

        def assignWordProbabilities(self):
            if smoothingParam == 0:
                self.unknownWordProbability = float('-inf') # Because log(0) is negative infinity
            else:
                self.unknownWordProbability = math.log(smoothingParam/(self.WordsSeen + smoothingParam*self.word_probs.__len__()))

            for w in self.word_probs:
                self.word_probs[w].prob = math.log((self.word_probs[w].count + smoothingParam)/(self.WordsSeen + smoothingParam*self.word_probs.__len__()))

    def tag(self, sentence):
        tagList = list(self.tags.keys()) # Get a list of the tags for the end
        # sentence = word_tokenize(sentence) # Turn sentence string into tokens
        sentence = tknzr.tokenize(sentence)

        # Create Path Prob and BackPointer Matrices
        n = self.tags.__len__()
        m = sentence.__len__()
        viterbi = [0]*n
        backpointer = [0]*n
        for i in range(n):
            viterbi[i] = [0]*m
            backpointer[i] = [0]*m

        # Initialization Step, Fill the First Column
        for counter, i in enumerate(self.tags):
            # print(self.tags[i])
            viterbi[counter][0] = self.a('<s>', i) + self.b(i, sentence[0]) # Add b/c Logs
            backpointer[counter][0] = None

        # Recursion Step, Fill the Rest of the Columns
        for t in range(1,m): # Goes from 1 or Col2 TO m-1 or last column
            for counter, i in enumerate(self.tags):
                the_max = float('-inf')
                maxBackPoint = None

                for arg_counter, arg_i in enumerate(self.tags):
                    test = viterbi[arg_counter][t-1] + self.a(arg_i, i) + self.b(i, sentence[t])
                    if test > the_max:
                        the_max = test
                        maxBackPoint = arg_counter

                viterbi[counter][t] = the_max
                backpointer[counter][t] = maxBackPoint

        # Termination Step
        viterbi_final = float('-inf')
        backpointer_final = None

        for counter, i in enumerate(self.tags):
            test = viterbi[counter][m-1] + self.a(i, '</s>')
            if test > viterbi_final:
                viterbi_final = test
                backpointer_final = counter

        output_tags = []
        pointer = backpointer_final

        # TheFinalTag
        # print(tagList[pointer])
        output_tags.append(tagList[pointer])

        for tt in range(-1,-1*m-1,-1):
            pointer = backpointer[pointer][tt] # pointer has old value that directs it to grab next pointer and assign to self
            if tt == -1*m or pointer == None:
                # print('tt: ', tt, ' ',tt == -1*m, ' and pointer: ', pointer)
                print('Sentence Tagged.')
            else:
                output_tags.append(tagList[pointer])

        output_tags.reverse()
        return output_tags

    def a(self, prevTag, currentTag):
        if currentTag not in self.tags:
            print('Error: Cannot find current tag in list of tags. (a function)')
        if prevTag not in self.tags[currentTag].transition_probs:
            return float('-inf')
        else:
            return self.tags[currentTag].transition_probs[prevTag].prob

    def b(self, currentTag, currentWord):
        if currentTag not in self.tags:
            print('Error: Cannot find current tag in list of tags. (a function)')
        if currentWord not in self.tags[currentTag].word_probs:
            return self.tags[currentTag].unknownWordProbability
        else:
            return self.tags[currentTag].word_probs[currentWord].prob

# Main Program and UI

print('Welcome to my POS Tagging Algorithm. Please wait while I learn some probabilities...')
# testing = input('Enter: ')
my_tagset = tagset()

w = brown.tagged_sents()

for counter, p in enumerate(w._pieces):

    print('Status: ', counter * 100 / w._pieces.__len__(), '%')
    for sent in p:
        # Sentence Starter
        my_tagset.updateTagset('<s>', '<s>', None)
        previous_tag = '<s>'

        # Actual Sentence
        for tup in sent:
            # print(tup[0])
            # print(tup[1])

            my_tagset.updateTagset(tup[1], tup[0], previous_tag)
            previous_tag = tup[1]

        # End of Sentence
        my_tagset.updateTagset('</s>', '</s>', previous_tag)

    # For Now to Test

my_tagset.assignProbabilities()
print('Done Training!')
string_to_tag = input('Enter some sentence(s) to tag: ')
sents_to_tag = sent_tokenize(string_to_tag)
for sentence in sents_to_tag:
    print('Program: ', my_tagset.tag(sentence))
    print('NLTK Result: ', nltk.pos_tag(word_tokenize(sentence)))

