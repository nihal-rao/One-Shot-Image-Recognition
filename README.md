# siamese-nn

An implementation of the paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

Differences from the paper:
* Same learning rate for all parameters, instead of layer wise learning rate as in the paper.
* No learning rate decay.

Usage:
* Download the Omniglot 'background' and 'evaluation' zip files from [here](https://github.com/brendenlake/omniglot).
* Run `data_prep.py` to split the dataset into train-val-test:
 ```
 python data_prep.py <DATA_DIR>
 ```
 where `DATA_DIR` is the directory containing both the zip files.
 * Run train.py to train the model:
```
python train.py <DATA_DIR>
```
* Hyperparameters such as learning rate, number of pairs fpr training, use of augmentation etc. can be changed in `train.py` .
