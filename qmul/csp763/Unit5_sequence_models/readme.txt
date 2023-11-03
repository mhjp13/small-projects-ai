This Jupyter Notebook is modified from [https://www.cse.chalmers.se/~richajo/nlp2019/l6/Sequence%20tagging%20example.html]
,and be modified it according to 
[https://www.cse.chalmers.se/~richajo/nlp2019/l6/Sequence%20tagging%20example.html]
[https://stackoverflow.com/questions/66516388/attributeerror-module-torchtext-data-has-no-attribute-field]

Running enviroment:
Google colab,GPU
or
https://jhub.eecs.qmul.ac.uk/ with python 3.9 (on the ECS763 Natural Language Processing server instance)

Data:
Note:Two packages of training and validation data sets are given. You can use any package.
eng.train&eng.test(original Conll03 corpus) eng.train.openNLP&eng.test.openNLP(retagging of Conll03 with openNLP,there are many tags that differ from the original Conll03 corpus: estimation at 12% of different tags)

Model:
RNN+CRF:RNNCRFTagger
RNN+Linear:RNNTagger