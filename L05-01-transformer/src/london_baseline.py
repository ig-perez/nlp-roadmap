# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import csv
import utils

eval_corpus_path = "birth_dev.tsv"

file = open(eval_corpus_path)
reader = csv.reader(file)
num_lines = len(list(reader))
file.close()

total, correct = utils.evaluate_places(eval_corpus_path, ["London"]*num_lines)

if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))