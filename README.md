# TAC-KBP-2017-Participation
Event Nuggest Detection and Coreference Resolution

Replicating our results submitted to the TAC KBP 2017 Event Nugget Detection and Coreference Evaluation.
=======================================================================================================
1) Store all files (coreNLP output) in data/2017/out folder.
2) Download pretrained models from https://drive.google.com/open?id=0B1ihlo1F9aKOdF9nYWlLWWkzTHc and store them in codes/type_models and codes/realis_models 
3) Run ensemble_classifier_test.py
===================================================================================
========Mention Type Results==========
                        Type	Prec	Rec	F1	#Gold	#Sys
              conflictattack	0.29	0.27	0.28	499	468
         conflictdemonstrate	0.52	0.62	0.57	189	225
            contactbroadcast	0.44	0.32	0.37	581	426
              contactcontact	0.29	0.42	0.35	446	648
       contactcorrespondence	0.15	0.09	0.11	96	61
                 contactmeet	0.35	0.41	0.38	147	176
           justicearrestjail	0.48	0.65	0.56	92	124
                     lifedie	0.56	0.55	0.55	233	230
                  lifeinjure	0.73	0.73	0.73	37	37
         manufactureartifact	0.32	0.19	0.24	112	66
   movementtransportartifact	0.10	0.02	0.04	123	30
     movementtransportperson	0.27	0.27	0.27	444	447
              personnelelect	0.24	0.19	0.21	86	66
        personnelendposition	0.48	0.52	0.50	186	203
      personnelstartposition	0.15	0.13	0.14	72	64
      transactiontransaction	0.03	0.06	0.04	36	60
    transactiontransfermoney	0.26	0.21	0.23	477	379
transactiontransferownership	0.29	0.22	0.25	299	225

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
