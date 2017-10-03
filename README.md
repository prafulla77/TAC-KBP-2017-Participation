# TAC-KBP-2017-Participation
Event Nuggest Detection and Coreference Resolution

Replicating our results submitted to the TAC KBP 2017 Event Nugget Detection and Coreference Evaluation.
=======================================================================================================
1) Store all files (coreNLP output) in data/2017/out folder.
2) Download pretrained models from https://drive.google.com/open?id=0B1ihlo1F9aKOdF9nYWlLWWkzTHc and store them in codes/type_models and codes/realis_models 
3) Run ensemble_classifier_test.py
===================================================================================
Result on TAC KBP 2016 Evaluation Data:
=======Final Mention Detection Results=========
                          	     Micro Average	     Macro Average
                Attributes	Prec  	Rec  	F1   	Prec  	Rec  	F1   
                     plain	56.97	53.96	55.42	53.86	52.23	53.03
              mention_type	46.02	43.58	44.77	42.57	41.63	42.09
             realis_status	43.27	40.98	42.09	40.46	40.04	40.24
mention_type+realis_status	34.32	32.50	33.39	31.22	31.12	31.17

=======Final Mention Coreference Results=========
Metric : bcub	Score	35.36
Metric : ceafe	Score	33.88
Metric : ceafm	Score	34.88 *
Metric : muc	Score	19.36
Metric : blanc	Score	19.34
Overall Average CoNLL score	26.99
