# TAC-KBP-2017-Participation
Event Nuggest Detection and Coreference Resolution

In case you use this code, please cite the following paper:

```
@inproceedings{choubey2017tamu,
  title={TAMU at KBP 2017: Event Nugget Detection and Coreference Resolution},
  author={Choubey, Prafulla Kumar and Huang, Ruihong},
  booktitle = {Text Analysis Conference (TAC 2017)},
  year = {2017}
}
```

Replicating our results submitted to the TAC KBP 2017 Event Nugget Detection and Coreference Evaluation.
=======================================================================================================
1) Store all files (coreNLP output) in data/2017/out folder.
2) Download pretrained models from https://drive.google.com/open?id=0B1ihlo1F9aKOdF9nYWlLWWkzTHc and store them in codes/type_models and codes/realis_models 
3) Run ensemble_classifier_test.py
===================================================================================
# Result on KBP 2016 Evaluation data
=======Final Mention Detection Results=========

                          	     Micro Average	     Macro Average
                                 
                Attributes	Prec  	Rec  	F1   	Prec  	Rec  	F1   
                     plain	52.38	60.48	56.14	49.82	58.97	54.01
              mention_type	41.50	47.92	44.48	38.80	45.93	42.06
             realis_status	39.74	45.88	42.59	37.39	44.97	40.83
             
mention_type+realis_status	30.79	35.55	33.00	28.27	34.05	30.89

=======Final Mention Coreference Results=========

Metric : bcub	Score	36.62

Metric : ceafe	Score	35.50

Metric : ceafm	Score	35.90 *

Metric : muc	Score	17.62

Metric : blanc	Score	18.77

Overall Average CoNLL score	27.13
