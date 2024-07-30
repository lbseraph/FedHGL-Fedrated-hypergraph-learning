# code for "Federated Hypergraph Learning with Hyperedge Completion"

## Description:

This repository contains code to reproduce "Federated Hypergraph Learning with Hyperedge Completion". This paper introduces a novel federated hypergraph learning method: FedHGN. It can effectively address the challenges of distributed training of high-order graph data. Experimental results on real-world datasets confirm FedHGN’s effectiveness and its competitive edge over existing federated subgraph learning methods.

## Datasets

* hypergraph datasets: Cora-CA, DBLP4k, IMDB4k, 20News
* simple graph: Cora, CiteSeer, Facebook

## How to Run

1.Install Python 3.9 and the necessary dependencies with the command ``pip install -r requirements.txt``.

2.Select from seven datasets(Cora-CA, etc.) and set the number of clients.

3.Run the code. (See detailed examples below). On hypergraph datasets: e.g. ``python ./src/main.py --dname cora-ca --num_layers 2 --n_client 1 --method FedHGN --global_rounds 600`` Use 1 client to perform 600 rounds on the cora-ca dataset to reproduce the experimental results. On simple graph datasets: e.g. ``python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedGCN`` Use 3 clients to reproduce the experimental results using the FedGCN algorithm on the cora dataset.

## Our experimental results are listed below for reference:

### On hypergraph datasets:

```
[cora-ca]

python ./src/main.py --dname cora-ca --num_layers 2 --n_client 1 --method FedHGN  --global_rounds 600 
Final Test Accuracy:  0.6953 0.0216

(c 3)

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 3  --method FedHGN 
Final Test Accuracy:  0.5353 0.0234

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 3  --method FedHGN --HC
Final Test Accuracy:  0.6369 0.0262

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 3  --method FedHGN 
Final Test Accuracy:  0.5831 0.0229

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 3  --method FedHGN   --HC
Final Test Accuracy:  0.69 0.0248

(c 6)

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 6  --method FedHGN 
Final Test Accuracy:  0.3778 0.0226

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 6  --method FedHGN -HC
Final Test Accuracy:  0.5545 0.0297

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 6  --method FedHGN 
Final Test Accuracy:  0.4703 0.0322

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 6  --method FedHGN    --HC
Final Test Accuracy:  0.6726 0.0282

(c 9)

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 9  --method FedHGN 
Final Test Accuracy:  0.3155 0.025

python ./src/main.py --dname cora-ca --num_layers 2  --local --n_client 9  --method FedHGN --HC
Final Test Accuracy:  0.5104 0.0231

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 9  --method FedHGN 
Final Test Accuracy:  0.3815 0.0246

python ./src/main.py --dname cora-ca --num_layers 2  --n_client 9  --method FedHGN    --HC
Final Test Accuracy:  0.6585 0.0291
```

```
[dblp4k]

python ./src/main.py --dname dblp --num_layers 2 --n_client 1 --method FedHGN  --global_rounds 600 --train_ratio 0.06
Final Test Accuracy:  0.8515 0.0187

(c 3)

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 3  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.7643 0.0229

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 3  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.781 0.0265

python ./src/main.py --dname dblp --num_layers 2  --n_client 3  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.7734 0.0443

python ./src/main.py --dname dblp --num_layers 2  --n_client 3  --method FedHGN   --HC --train_ratio 0.06
FFinal Test Accuracy:  0.7859 0.041

(c 6)

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 6  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.7006 0.0274

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 6  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.7065 0.0351

python ./src/main.py --dname dblp --num_layers 2  --n_client 6  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.7263 0.062

python ./src/main.py --dname dblp --num_layers 2  --n_client 6  --method FedHGN  --HC --train_ratio 0.06
Final Test Accuracy:  0.7481 0.0606

(c 9)

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 9  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.647 0.0268

python ./src/main.py --dname dblp --num_layers 2  --local --n_client 9  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.6529 0.0352

python ./src/main.py --dname dblp --num_layers 2  --n_client 9  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.7083 0.04

python ./src/main.py --dname dblp --num_layers 2  --n_client 9  --method FedHGN  --HC --train_ratio 0.06
Final Test Accuracy:  0.7249 0.0407
```

```
[imdb4k]

python ./src/main.py --dname imdb --num_layers 2 --n_client 1 --method FedHGN --global_rounds 600 --train_ratio 0.06
Final Test Accuracy:  0.5413 0.0205

(c 3)

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 3  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.4379 0.0181

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 3  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.4882 0.0224

python ./src/main.py --dname imdb --num_layers 2  --n_client 3  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.5292 0.02

python ./src/main.py --dname imdb --num_layers 2  --n_client 3  --method FedHGN  --HC --train_ratio 0.06
Final Test Accuracy:  0.5329 0.0198


(c 6)

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 6  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.3758 0.0135

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 6  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.4466 0.0201

python ./src/main.py --dname imdb --num_layers 2  --n_client 6  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.4088 0.0195

python ./src/main.py --dname imdb --num_layers 2  --n_client 6  --method FedHGN  --HC --train_ratio 0.06
Final Test Accuracy:  0.5325 0.0205

(c 9)

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 9  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.3481 0.0168

python ./src/main.py --dname imdb --num_layers 2  --local --n_client 9  --method FedHGN --HC --train_ratio 0.06
Final Test Accuracy:  0.4191 0.0179

python ./src/main.py --dname imdb --num_layers 2  --n_client 9  --method FedHGN --train_ratio 0.06
Final Test Accuracy:  0.3727 0.0167

python ./src/main.py --dname imdb --num_layers 2  --n_client 9  --method FedHGN  --HC --train_ratio 0.06
Final Test Accuracy:  0.5292 0.02
```

```
[news20]

python ./src/main.py --dname news --num_layers 2 --n_client 1 --method FedHGN --global_rounds 600 --train_ratio 0.01
Final Test Accuracy:  0.7888 0.0073

(c 3)

python ./src/main.py --dname news --num_layers 2  --local --n_client 3  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.763 0.012

python ./src/main.py --dname news --num_layers 2  --local --n_client 3  --method FedHGN --train_ratio 0.01 --HC
Final Test Accuracy:  0.7646 0.0135

python ./src/main.py --dname news --num_layers 2   --n_client 3  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.7832 0.0085

python ./src/main.py --dname news --num_layers 2  --n_client 3  --method FedHGN --train_ratio 0.01 --HC
Final Test Accuracy:  0.7843 0.0077

(c 6)

python ./src/main.py --dname news --num_layers 2  --local --n_client 6  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.7245 0.0131

python ./src/main.py --dname news --num_layers 2  --local --n_client 6  --method FedHGN --train_ratio 0.01 --HC
Final Test Accuracy:  0.7297 0.0123

python ./src/main.py --dname news --num_layers 2  --n_client 6  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.7755 0.0102

python ./src/main.py --dname news --num_layers 2  --n_client 6  --method FedHGN --train_ratio 0.01  --HC
Final Test Accuracy:  0.7793 0.0099

(c 9)

python ./src/main.py --dname news --num_layers 2  --local --n_client 9  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.7027 0.021

python ./src/main.py --dname news --num_layers 2  --local --n_client 9  --method FedHGN --train_ratio 0.01 --HC
Final Test Accuracy:  0.7076 0.0206

python ./src/main.py --dname news --num_layers 2  --n_client 9  --method FedHGN --train_ratio 0.01
Final Test Accuracy:  0.7699 0.0153

python ./src/main.py --dname news --num_layers 2  --n_client 9  --method FedHGN --train_ratio 0.01  --HC
Final Test Accuracy:  0.7746 0.0138
```

### On simple graph datasets:

```
[cora]

(K 3 gr 200)

python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedSage 
Final Test Accuracy:  0.6858 0.0242

python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedGCN 
Final Test Accuracy:  0.7272 0.019

python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedHGN 
Final Test Accuracy:  0.7565 0.0149

python ./src/main.py --dname cora --num_layers 2  --n_client 3 --method FedSage --HC 
Final Test Accuracy:  0.812 0.0171

python ./src/main.py --dname cora --num_layers 2  --n_client 3 --method FedGCN  --HC 
Final Test Accuracy:  0.8277 0.0139

python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedCog 
Final Test Accuracy:  0.8178 0.0199

python ./src/main.py --dname cora --num_layers 2 --n_client 3 --method FedHGN  --HC 
Final Test Accuracy:  0.8352 0.0205

(K 8 gr 200)

python ./src/main.py --dname cora --num_layers 2 --n_client 8 --method FedSage 
Final Test Accuracy:  0.5637 0.028

python ./src/main.py --dname cora --num_layers 2 --n_client 8 --method FedGCN 
Final Test Accuracy:  0.6105 0.0339

python ./src/main.py --dname cora --num_layers 2 --n_client 8 --method FedHGN 
Final Test Accuracy:  0.6017 0.0253

python ./src/main.py --dname cora --num_layers 2  --n_client 8 --method FedSage  --HC 
Final Test Accuracy:  0.7959 0.0175

python ./src/main.py --dname cora --num_layers 2  --n_client 8 --method FedGCN  --HC 
Final Test Accuracy:  0.8107 0.0211

python ./src/main.py --dname cora --num_layers 2 --n_client 8 --method FedCog 
Final Test Accuracy:  0.8226 0.0199

python ./src/main.py --dname cora --num_layers 2  --n_client 8 --method FedHGN  --HC 
Final Test Accuracy:  0.8318 0.017
```

```
[citeseer]

(c 3 gr 200)

python ./src/main.py --dname citeseer --num_layers 2 --n_client 3 --method FedSage 
Final Test Accuracy:  0.6301 0.0167

python ./src/main.py --dname citeseer --num_layers 2 --n_client 3 --method FedGCN 
Final Test Accuracy:  0.6563 0.0175

python ./src/main.py --dname citeseer --num_layers 2 --n_client 3 --method FedHGN 
Final Test Accuracy:  0.5849 0.0212

python ./src/main.py --dname citeseer --num_layers 2  --n_client 3  --method FedSage  --HC 
Final Test Accuracy:  0.6893 0.0196

python ./src/main.py --dname citeseer --num_layers 2  --n_client 3  --method FedGCN  --HC 
Final Test Accuracy:  0.6997 0.0192

python ./src/main.py --dname citeseer --num_layers 2  --n_client 3  --method FedCog 
Final Test Accuracy:  0.7034 0.0154

python ./src/main.py --dname citeseer --num_layers 2  --n_client 3  --method FedHGN  --HC 
Final Test Accuracy:  0.7076 0.0139

(c 8 gr 200)

python ./src/main.py --dname citeseer --num_layers 2 --n_client 8 --method FedSage 
Final Test Accuracy:  0.598 0.0215

python ./src/main.py --dname citeseer --num_layers 2 --n_client 8 --method FedGCN 
Final Test Accuracy:  0.6159 0.0224

python ./src/main.py --dname citeseer --num_layers 2 --n_client 8 --method FedHGN 
Final Test Accuracy:  0.4335 0.0223

python ./src/main.py --dname citeseer --num_layers 2  --n_client 8  --method FedSage  --HC 
Final Test Accuracy:  0.6818 0.0192

python ./src/main.py --dname citeseer --num_layers 2  --n_client 8  --method FedGCN  --HC 
Final Test Accuracy:  0.697 0.0143

python ./src/main.py --dname citeseer --num_layers 2  --n_client 8  --method FedCog 
Final Test Accuracy:  0.7137 0.0133

python ./src/main.py --dname citeseer --num_layers 2  --n_client 8  --method FedHGN  --HC 
Final Test Accuracy:  0.7171 0.0174
```

```
[facebook]

(c 3 gr 200)

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedSage   --train_ratio 0.008
Final Test Accuracy:  0.7141 0.0126

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedGCN   --train_ratio 0.008
Final Test Accuracy:  0.7348 0.0182

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedHGN   --train_ratio 0.008
Final Test Accuracy:  0.8046 0.0084

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedSage  --HC --train_ratio 0.008
Final Test Accuracy:  0.7935 0.0134

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedGCN  --HC --train_ratio 0.008
Final Test Accuracy:  0.8219 0.0108

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedCog   --train_ratio 0.008
Final Test Accuracy:  0.8078 0.0105

python ./src/main.py --dname facebook --num_layers 2  --n_client 3  --method FedHGN  --HC --train_ratio 0.008
Final Test Accuracy:  0.8409 0.0092

(c 8 gr 200)

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedSage   --train_ratio 0.008
Final Test Accuracy:  0.6025 0.0188

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedGCN   --train_ratio 0.008
Final Test Accuracy:  0.6455 0.0091

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedHGN   --train_ratio 0.008
Final Test Accuracy:  0.7325 0.0122

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedSage  --HC --train_ratio 0.008
Final Test Accuracy:  0.7683 0.0118

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedGCN  --HC --train_ratio 0.008
Final Test Accuracy:  0.8166 0.0098

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedCog   --train_ratio 0.008
Final Test Accuracy:  0.8116 0.0131

python ./src/main.py --dname facebook --num_layers 2  --n_client 8  --method FedHGN  --HC --train_ratio 0.008
Final Test Accuracy:  0.8395 0.0128
```
