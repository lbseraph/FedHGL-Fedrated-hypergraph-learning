import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # Set the font type
    "axes.unicode_minus": False, # Solve the problem that the negative sign cannot be displayed
    'font.size': 16
}
rcParams.update(config)

y1=[0.38040665,0.3784658,0.37171904,0.3689464,0.36728281,0.36737523
,0.36829945,0.37079482,0.37707948,0.38401109,0.39334566,0.40619224
,0.42449168,0.44528651,0.46515712,0.4849353,0.50332717,0.52227357
,0.5349353,0.55073937,0.56090573,0.57430684,0.58669131,0.59445471
,0.6012939,0.6103512,0.6181146,0.62615527,0.63123845,0.63521257
,0.63817006,0.64020333,0.64195933,0.64417745,0.64510166,0.64778189
,0.64916821,0.65027726,0.65332717,0.65591497,0.65794824,0.65859519
,0.66016636,0.66192237,0.66201479,0.66321627,0.66358595,0.66404806
,0.66497227,0.66626617,0.66756007,0.66866913,0.66848429,0.66987061
,0.67088725,0.6707024,0.6715342,0.67255083,0.67171904,0.67301294
,0.67338262,0.67365989,0.674122,0.67467652,0.67550832,0.67652495
,0.67754159,0.67865065,0.68012939,0.6805915,0.67985213,0.68003697
,0.68022181,0.6810536,0.68114603,0.68216266,0.68133087,0.68114603
,0.68151571,0.6823475,0.68160813,0.68207024,0.68207024,0.68179298
,0.68271719,0.68299445,0.68354898,0.68512015,0.68438078,0.68475046
,0.68465804,0.68401109,0.68438078,0.68521257,0.68465804,0.68604436
,0.68604436,0.6857671,0.68558226,0.68558226,0.68548983,0.68622921
,0.68613678,0.68678373,0.68724584,0.68604436,0.68641405,0.68715342
,0.68678373,0.68678373,0.68715342,0.68789279,0.68798521,0.68890943
,0.68881701,0.68890943,0.68844732,0.68872458,0.69029575,0.69066543
,0.69112754,0.69057301,0.68946396,0.68872458,0.68918669,0.68955638
,0.68872458,0.68826248,0.68927911,0.69001848,0.69066543,0.69149723
,0.69048059,0.69057301,0.69066543,0.69177449,0.69158965,0.69121996
,0.69048059,0.69075786,0.69038817,0.69029575,0.69057301,0.69066543
,0.6909427,0.69085028,0.69121996,0.69149723,0.69260628,0.69149723
,0.69085028,0.69103512,0.6909427,0.69112754,0.69066543,0.69038817
,0.69048059,0.6909427,0.69057301,0.69075786,0.69112754,0.69186691
,0.69121996,0.69112754,0.69140481,0.69279113,0.69251386,0.69168207
,0.69186691,0.69131238,0.69149723,0.69205176,0.69214418,0.69140481
,0.69057301,0.69029575,0.69131238,0.69232902,0.69297597,0.69260628
,0.69214418,0.69269871,0.69242144,0.69260628,0.69158965,0.69195933
,0.69029575,0.69048059,0.69214418,0.69158965,0.69103512,0.69066543
,0.69066543,0.69029575,0.69075786,0.69214418,0.69242144,0.69251386
,0.69177449,0.69177449,0.69011091,0.69205176,0.69297597,0.69325323
,0.69297597,0.69232902,0.69140481,0.6909427,0.69121996,0.69177449
,0.6922366,0.69297597,0.69306839,0.6922366,0.69297597,0.6909427
,0.69131238,0.69112754,0.69186691,0.69288355,0.69279113,0.69242144
,0.69269871,0.69288355,0.69168207,0.6922366,0.69177449,0.69195933
,0.69242144,0.69380776,0.69408503,0.69390018,0.69371534,0.69242144
,0.69269871,0.69186691,0.69177449,0.69306839,0.69214418,0.69121996
,0.69140481,0.69103512,0.69057301,0.69195933,0.69158965,0.69103512
,0.69103512,0.69195933,0.69316081,0.69306839,0.69297597,0.69186691
,0.69168207,0.69260628,0.69140481,0.69103512,0.69121996,0.69140481
,0.69269871,0.69232902,0.69140481,0.69260628,0.69343808,0.69325323
,0.69399261,0.69417745,0.69371534,0.69297597,0.6922366,0.69112754
,0.69112754,0.69186691,0.69112754,0.69195933,0.69316081,0.69325323
,0.69242144,0.69186691,0.69048059,0.69085028,0.69121996,0.69186691
,0.69260628,0.69251386,0.69195933,0.69205176,0.69251386,0.69269871
,0.69195933,0.69371534,0.69232902,0.69232902,0.69260628,0.69279113
,0.69242144,0.69242144,0.6909427,0.69186691,0.69260628,0.69269871
,0.69177449,0.69279113,0.69279113,0.69168207,0.69131238,0.69168207
,0.69103512,0.6922366,0.6922366,0.69251386,0.69288355,0.69186691
,0.69075786,0.68983364,0.69112754,0.6922366,0.69242144,0.69306839
,0.69334566,0.69260628,0.69297597,0.6909427,0.69242144,0.69205176
,0.69158965,0.69149723,0.69214418,0.69205176,0.69205176,0.69205176
,0.69112754,0.69168207,0.69251386,0.69186691,0.69186691,0.6909427
,0.69085028,0.69020333,0.69075786,0.69075786,0.69158965,0.69195933
,0.69242144,0.69205176,0.69260628,0.69205176,0.69251386,0.69186691
,0.69177449,0.69316081,0.69454713,0.69399261,0.6935305,0.69297597
,0.69316081,0.69251386,0.69316081,0.69325323,0.69362292,0.69371534
,0.69343808,0.69380776,0.69251386,0.69279113,0.69205176,0.69112754
,0.69168207,0.69371534,0.69269871,0.69334566,0.69177449,0.69066543
,0.69112754,0.69205176,0.69251386,0.69251386,0.69297597,0.69149723
,0.69131238,0.69168207,0.69195933,0.69251386,0.69214418,0.69242144
,0.69279113,0.69195933,0.69085028,0.69131238,0.69121996,0.69232902
,0.69306839,0.69334566,0.69316081,0.69205176,0.69260628,0.69279113
,0.69214418,0.69140481,0.68918669,0.69057301,0.69195933,0.69168207
,0.69186691,0.69168207,0.69214418,0.69168207,0.69186691,0.69242144
,0.69232902,0.69158965,0.69075786,0.69260628,0.69112754,0.69214418
,0.69195933,0.69205176,0.69242144,0.69316081,0.69334566,0.69232902
,0.69362292,0.69279113,0.69232902,0.69251386,0.69214418,0.69121996
,0.69066543,0.69011091,0.69112754,0.69177449,0.69195933,0.69306839
,0.69297597,0.69214418,0.69242144,0.69251386,0.69242144,0.6935305
,0.69279113,0.69242144,0.6922366,0.69149723,0.69103512,0.69121996
,0.69158965,0.69232902,0.69242144,0.69232902,0.69316081,0.6935305
,0.69334566,0.69316081,0.69214418,0.69158965,0.69158965,0.69214418
,0.69195933,0.6909427,0.69121996,0.69186691,0.6922366,0.69195933
,0.69214418,0.69195933,0.69158965,0.69140481,0.69149723,0.69131238
,0.69029575,0.69066543,0.69242144,0.69205176,0.69195933,0.69260628
,0.69177449,0.69232902,0.69121996,0.69214418,0.69269871,0.6935305
,0.69279113,0.69325323,0.69260628,0.69214418,0.69288355,0.69316081
,0.69371534,0.69362292,0.69408503,0.69408503,0.69325323,0.69288355
,0.69158965,0.69205176,0.69251386,0.69112754,0.6922366,0.69168207
,0.69205176,0.69306839,0.69306839,0.69131238,0.69205176,0.69103512
,0.69103512,0.69158965,0.69279113,0.69260628,0.69214418,0.69140481
,0.69149723,0.6922366,0.69057301,0.69085028,0.69038817,0.69186691
,0.69242144,0.69316081,0.69288355,0.69288355,0.69232902,0.69232902
,0.69232902,0.69112754,0.69260628,0.69288355,0.69186691,0.6922366
,0.69195933,0.69168207,0.69288355,0.69269871,0.69186691,0.69186691
,0.69158965,0.69325323,0.69177449,0.69306839,0.69242144,0.69195933
,0.69269871,0.69242144,0.69214418,0.69195933,0.69103512,0.69103512
,0.69186691,0.69214418,0.69195933,0.69168207,0.69168207,0.69186691
,0.69112754,0.69048059,0.69075786,0.69149723,0.69214418,0.69214418
,0.69288355,0.69316081,0.69297597,0.69371534,0.69297597,0.69334566
,0.69242144,0.69140481,0.69177449,0.69325323,0.69288355,0.69269871
,0.69297597,0.69269871,0.69251386,0.69232902,0.6922366,0.69232902
,0.69269871,0.69205176,0.69186691,0.69205176,0.69232902,0.69232902
,0.6935305,0.69186691,0.69140481,0.69242144,0.69297597,0.69380776
,0.69232902,0.69158965,0.69158965,0.69103512,0.69195933,0.69279113
,0.69242144,0.6922366,0.69279113,0.69325323,0.69306839,0.69288355]


y2=[0.29986048,0.31487921,0.31645242,0.31654501,0.31784268,0.31904793
,0.32210675,0.3229406,0.32368255,0.3243319,0.32498143,0.32674327
,0.32850613,0.33138097,0.33397821,0.33722325,0.34260135,0.34528998
,0.34992597,0.35307962,0.35456402,0.35688279,0.35994264,0.3620764
,0.36356029,0.36467363,0.3660651,0.36597113,0.36643564,0.36810558
,0.36986621,0.36995898,0.37051505,0.37116457,0.37209256,0.37237017
,0.37264812,0.37274157,0.37339058,0.37366784,0.37404045,0.37469014
,0.3746886,0.37440979,0.37394459,0.37366612,0.37375871,0.37413029
,0.3738532,0.37366818,0.37431667,0.37440944,0.37422339,0.37385371
,0.37422459,0.37366766,0.37357421,0.37357352,0.37357421,0.37394596
,0.37459411,0.37459376,0.37440806,0.37468653,0.37487188,0.37477878
,0.37496482,0.37505707,0.37422288,0.37459445,0.37422528,0.37441046
,0.37413166,0.37478101,0.37496706,0.3744103,0.37385336,0.37376146
,0.37422494,0.3743177,0.37487291,0.37505844,0.37505759,0.37524226
,0.37616956,0.37598403,0.37570557,0.37607662,0.37598386,0.37607576
,0.37570488,0.37579781,0.37607473,0.3756116,0.37561211,0.37626249
,0.37552141,0.37589178,0.37561365,0.37561245,0.37570573,0.37681873
,0.37681874,0.37691013,0.37691133,0.3764475,0.37570573,0.37598317
,0.37644733,0.37681787,0.37644716,0.37653924,0.37644699,0.37663201
,0.37616887,0.37635372,0.37681823,0.3770053,0.37672735,0.37737584
,0.37765311,0.37635526,0.37598403,0.37616939,0.375983,0.37607628
,0.37663372,0.37616887,0.37589092,0.37542795,0.37515035,0.37477878
,0.37440755,0.37468739,0.37524415,0.37607903,0.37570746,0.37635664
,0.37672752,0.37691254,0.3766351,0.37663321,0.3773755,0.37709737
,0.37718997,0.37709806,0.37746809,0.37691236,0.37663424,0.37644802
,0.37644836,0.37644819,0.37598402,0.37626335,0.37644957,0.37672837
,0.37719185,0.37700581,0.37756102,0.37802382,0.37830194,0.3784873
,0.37811641,0.37718911,0.37672494,0.37607628,0.37579816,0.37579815
,0.37654079,0.37654096,0.37672632,0.37626267,0.37616973,0.37496516
,0.37561486,0.37552261,0.37552244,0.37617179,0.37626404,0.37691236
,0.37626335,0.37598523,0.37598455,0.37635526,0.37616973,0.37589143
,0.37626232,0.3759842,0.37626232,0.37644768,0.37635509,0.37663355
,0.3762625,0.37598455,0.37561434,0.37561434,0.37607817,0.37617093
,0.37700667,0.37654199]


y3=[0.29986048,0.32757334,0.32608756,0.32024323,0.31755255,0.31217238
,0.31013294,0.30855508,0.30660926,0.30568196,0.30540487,0.30494259
,0.30559126,0.30661098,0.30828022,0.31050692,0.31236186,0.31523515
,0.31792515,0.32126279,0.32599085,0.3304408,0.33414898,0.33962071
,0.34499985,0.34843162,0.35371782,0.36020927,0.36493837,0.37022354
,0.3743038,0.37782645,0.3825552,0.38728585,0.39043949,0.39442648
,0.39748513,0.39980201,0.40239735,0.40378882,0.40657226,0.41018733
,0.41204348,0.41306372,0.4144545,0.41714346,0.41862786,0.41974069
,0.42215308,0.4234499,0.42511897,0.42651027,0.42762327,0.42929234
,0.43031275,0.43207407,0.43346485,0.43430059,0.43596949,0.4369887
,0.43837983,0.43921505,0.44097809,0.44171969,0.44227576,0.44320375
,0.44422399,0.44533613,0.44682002,0.44607791,0.44765421,0.44830322
,0.4482108,0.44895223,0.44932294,0.44950933,0.45062164,0.45117909
,0.4510865,0.45229209,0.4529411,0.45405444,0.45479639,0.45498226
,0.45535332,0.45563144,0.45590939,0.45646666,0.45702291,0.45720826
,0.45711516,0.45776468,0.45748622,0.45887751,0.4590627,0.45989809
,0.46054744,0.46073263,0.46091781,0.46064021,0.46064021,0.46073315
,0.46091953,0.46129042,0.46175407,0.4619396,0.46221704,0.46249533
,0.46314451,0.46360867,0.46370161,0.46333038,0.46332969,0.46286639
,0.46314416,0.46388662,0.46314485,0.46397939,0.4643501,0.46397939
,0.46444235,0.46379351,0.46397922,0.46425768,0.46490617,0.46536931
,0.4660178,0.46536862,0.46546087,0.46592418,0.46629489,0.46536776
,0.46536811,0.46499619,0.4656452,0.46555243,0.46573831,0.46629472
,0.46675889,0.46675889,0.46722151,0.46685079,0.46722168,0.4675936
,0.4681507,0.46833554,0.46777964,0.46750083,0.46796466,0.46805639
,0.46787103,0.46870625,0.46852004,0.46861332,0.46870677,0.46861383
,0.46880005,0.468614,0.46879936,0.46824295,0.46787258,0.46805776
,0.46944889,0.46935561,0.4687985,0.46824209,0.46851849,0.46842555
,0.46870505,0.46898386,0.469447,0.4687985,0.46870574,0.46944751
,0.46935475,0.46917025,0.46916991,0.46889264,0.46879937,0.46879902
,0.46907697,0.46935441,0.4690768,0.46981857,0.46963304,0.46926284
,0.46926301,0.46981926,0.46963424,0.47046895,0.47046912,0.47084053
,0.47028411,0.46982115,0.46954302,0.46935784,0.46954354,0.46963596
,0.46972752,0.46972718]

y4=[0.34056382,0.36520135,0.35629046,0.34923693,0.34311396,0.33745911
,0.33606713,0.33569607,0.33569555,0.33717789,0.33884714,0.34162904
,0.34524462,0.35043772,0.35627998,0.36555213,0.37566003,0.38919746
,0.40041887,0.41247362,0.42860605,0.44177586,0.45828416,0.47293717
,0.48452723,0.49343056,0.5031674,0.51234919,0.52180704,0.53070968
,0.53859191,0.54359707,0.54897517,0.55407533,0.56241932,0.56705702
,0.57243668,0.5773501,0.58096568,0.58430365,0.58680675,0.5899585
,0.59172137,0.59505986,0.59849111,0.60118163,0.60349988,0.60628161
,0.607486,0.60924852,0.61128883,0.61212439,0.61416521,0.61583497
,0.61778182,0.6189869,0.62047062,0.62213918,0.62269577,0.62334546
,0.62371703,0.62510644,0.62603323,0.62668309,0.62788938,0.62909514
,0.62983709,0.63048558,0.63094889,0.63150513,0.63270986,0.63326593
,0.63363613,0.63409979,0.6346562,0.63586128,0.63688135,0.63715929
,0.63827075,0.63836283,0.63919754,0.6402176,0.64086782,0.64142509
,0.64207375,0.64281467,0.64327849,0.64402061,0.6450405,0.6452262
,0.64606074,0.64661629,0.64763739,0.64837796,0.64837779,0.64828468
,0.64874833,0.64893369,0.6500472,0.65032636,0.65116141,0.65143902
,0.65051326,0.65218217,0.65227442,0.65310809,0.65422041,0.65468526
,0.65468371,0.65468423,0.65570464,0.65579741,0.65579827,0.65653987
,0.65663418,0.6565421,0.65700524,0.65765493,0.65737681,0.65746975
,0.65839722,0.65886225,0.65941866,0.65960436,0.65867775,0.65923468
,0.6596054,0.66025406,0.66090462,0.66118394,0.66136844,0.66127533
,0.66127447,0.66164536,0.66164519,0.66183089,0.66229454,0.66266473
,0.66285009,0.66294371,0.66433535,0.66415017,0.66424362,0.66480004
,0.66591287,0.66415,0.66535576,0.6658201,0.66637514,0.66683913
,0.66711794,0.6660058,0.66619168,0.66619116,0.66647031,0.66665773
,0.66712138,0.66786298,0.66814145,0.66860373,0.6695312,0.66925342
,0.6691598,0.66897461,0.6694386,0.66925308,0.66971741,0.66916083
,0.67027417,0.67036694,0.67138683,0.67092301,0.67055298,0.67166633
,0.67092473,0.67148252,0.67176064,0.67222447,0.67250208,0.67315091
,0.67305814,0.67250224,0.67352334,0.67352368,0.67343006,0.67352231
,0.67315246,0.67407873,0.67380095,0.67370835,0.67343041,0.67352317
,0.67380181,0.67287502,0.67361611,0.67389406,0.67444995,0.6744515
,0.67454478,0.67482256]

y5=[0.34056382,0.35632089,0.35798326,0.35612796,0.35769963,0.35946318
,0.36280167,0.36521115,0.36780873,0.37040459,0.37513352,0.37967726
,0.38718654,0.39423321,0.40081725,0.40916537,0.41751176,0.42622956
,0.43698474,0.44551581,0.4533039,0.46211498,0.47166543,0.47861693
,0.48418056,0.48844739,0.49215574,0.49632756,0.50059318,0.50309885
,0.5068096,0.50986945,0.51274378,0.51432043,0.51663902,0.51830947
,0.51821705,0.51960783,0.52164693,0.52192522,0.52229491,0.52322186
,0.5253539,0.52590894,0.52600016,0.52850377,0.52961643,0.53100858
,0.53165794,0.53276991,0.53314079,0.5336053,0.53425568,0.53453398
,0.53536851,0.53675826,0.53694328,0.53685,0.5372214,0.53712949
,0.53731519,0.53777988,0.53870701,0.53861458,0.5389853,0.54028331
,0.54120958,0.54102371,0.54037522,0.54046867,0.54009727,0.54065351
,0.54139563,0.54250863,0.54390061,0.54380802,0.54362214,0.54371387
,0.54343489,0.54445496,0.54491792,0.54547451,0.54519708,0.54510379
,0.54538106,0.5451945,0.54510259,0.54575263,0.54566038,0.54584591
,0.54658665,0.54760706,0.54649492,0.54677218,0.54705047,0.54732963
,0.54714256,0.5478845,0.54788467,0.54751447,0.54816262,0.54936719
,0.54899699,0.54862748,0.5488118,0.54890422,0.54825521,0.54936925
,0.54974099,0.55001825,0.55038794,0.54983307,0.54899733,0.54946063
,0.55029603,0.54992532,0.55168784,0.55205873,0.55261343,0.55233565
,0.55150128,0.55011033,0.55057587,0.54992583,0.55094676,0.55140955
,0.55187354,0.5517794,0.55168612,0.55122282,0.55038794,0.5499248
,0.54936873,0.55085262,0.55020344,0.55038983,0.55131713,0.55113263
,0.5510409,0.55122453,0.55159559,0.55206079,0.55215373,0.55085571
,0.55150403,0.55103987,0.55159645,0.55113229,0.55196631,0.55261463
,0.55382023,0.55428542,0.55428525,0.55354399,0.55354365,0.55373055
,0.5538228,0.55363676,0.55243082,0.55289395,0.55345175,0.55261669
,0.55206062,0.55206045,0.5525241,0.55270946,0.55391505,0.55335915
,0.55382263,0.55438008,0.55391694,0.5533607,0.55400867,0.5551215
,0.55447181,0.55437956,0.55447146,0.55456405,0.55354416,0.55363762
,0.55317328,0.5531731,0.55382297,0.5539166,0.55345381,0.55373055
,0.55335967,0.55345226,0.55317293,0.55354227,0.55307811,0.55335795
,0.55354433,0.55512202,0.5556774,0.55623296,0.55651142,0.55465631
,0.55437836,0.55419334]

if __name__ == '__main__':

    # plt.rcParams['font.sans-serif'] = [u'SimSun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.size'] = 14
    row_x = []
    for i in range(len(y1)):
        row_x.append(i)
    plt.plot(row_x, y1, label='Global HGNN', ls=':', c='grey')

    row_x = []
    for i in range(len(y2)):
        row_x.append(i)
    plt.plot(row_x, y2, label='Local HGNN', ls='-.', c='#ff7f0e')

    row_x = []
    for i in range(len(y5)):
        row_x.append(i)
    plt.plot(row_x, y5, label='Local HGNN \nwith HC', ls='-.', c='c')


    row_x = []
    for i in range(len(y3)):
        row_x.append(i)
    plt.plot(row_x, y3, label='FedHGN w/o HC', ls='--', c='#2ca02c')

    row_x = []
    for i in range(len(y4)):
        row_x.append(i)
    plt.plot(row_x, y4, label='FedHGN with HC', ls='-', c='#1f77b4')


    plt.ylim(0, 0.7)
    plt.xlim(0, 150)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    # plt.ylabel('故障分类准确率(%)')
    # plt.xlabel('训练迭代周期')
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.rcParams['savefig.dpi'] = 300 # Image pixels
    plt.show()
