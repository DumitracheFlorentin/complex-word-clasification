# Complex Word Clasification

## Documentation

Predicting which words are considered hard to understand for a given target population is a vital step in many Natural Language Processing applications such as text simplification. A system could simplify texts for second language learners, native speakers with low literacy levels, and people with reading disabilities. This task is commonly referred to as Complex Word Identification. Usually, this task is approached as a binary classification task in which systems predict a complexity value (complex vs. non-complex) for a set of target words in a text. In this challenge, the task is to predict the lexical complexity of a word in a sentence. A word which is considered to be complex has label 1, a word is considered to be simple (non-complex) has label 0.

## Evaluation

The evaluation metric for this competition is balanced accuracy score. This is a proper metric to compute the accuracy in binary classification problems where the dataset is imbalanced. It is defined as the average of recall obtained on each class:

balanced accuracy=
sensitivity+specificity
2
 
where

sensitivity=true positive rate = TP / (TP + FN)
 
specifity=true negative rate = TN / (FP + TN)
 
You can read details about the measure here: https://en.wikipedia.org/wiki/Sensitivity_and_specificity.

In the binary case, balanced accuracy is equal to the area under the ROC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve with binary predictions.

## Solution

This solution managed to obtain a performance of 0.805%.
