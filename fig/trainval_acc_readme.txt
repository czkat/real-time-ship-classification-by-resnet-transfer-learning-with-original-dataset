1. batch size 32, epoch 5, lr 1e-4
2. batch size 32, epoch 20
3. batch size 32, epoch 100
4. bigger dataset size, no parameter change
5: batch size 32, epoch 200, lr 1e-4=0.0001 --> overfitting (same as 4) 
6. augment with data x4 then split randomly: both 90% but train has leakage to test.
7. split as usual, then augment but data same size: no improvement --> overfitting
8. split as usual, then augment with data x4 in train and test seperately  
9. same as 7 but resnet34 to deal with overfitting 
10. resnet18
11-17. lr tuning at resnet18, best at lr 0.02 val acc 91.7. 11: 0.0001
12. 0.001
13. 0.01
14. 0.1
15. 0.08
16. 0.02
17. 0.05
18. updated, balanced class each 100 to 150, lr 0.02, resnet18, momentum 0.9, batchsize 32, epoch 200, val acc 
   lr 0.02 batchsize 32 layer 18: 94.73
19 lr 0.1  batchsize 32 layer 18: 83.0
20 lr 0.05 batchsize 32 layer 18: 92.7
21 lr 0.01 batchsize 32 layer 18: 93.9
22 lr0.001 batchsize 32 layer 18: 88.3 stable
23 lr0.0001batchsize 32 layer 18: 88.9 stable
24 lr 0.01 batchsize 64 layer 18: 92.2
25 lr 0.02 batchsize 64 layer 18: 93.7
26 lr 0.01 batchsize 32 layer 34: 94.8
27 lr 0.02 batchsize 32 layer 34: 96.7 stable





5 train 99 val 81
6 train 99 val 97
7 train 95 val 80
8 train 100 val 85
9 train 97.6 val 82.7
10 train 94 val 84

7 9 10: val highest at resnet18
7 8: good to aug with x4 data
5 7: train drop, val drop less: good to aug in general for better test acc

8. Epoch 200
Epoch [200/200], Step [0/60], Loss: 0.0334
Epoch [200/200], Step [20/60], Loss: 0.0310
Epoch [200/200], Step [40/60], Loss: 0.0165
train-loss: 0.2457, train-acc: 99.9477
validation loss: 0.5998, validation acc: 85.0309

9. Epoch [200/200], Step [0/15], Loss: 0.1307
train-loss: 0.5870, train-acc: 97.6987
validation loss: 0.8277, validation acc: 82.7160

10. Epoch [200/200], Step [0/15], Loss: 0.2118
train-loss: 0.7150, train-acc: 94.1423
validation loss: 0.8401, validation acc: 84.2593

learning rate tuning:
11 (acc_18)
Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0534
Epoch [200/200], Step [20/60], Loss: 0.0246
Epoch [200/200], Step [40/60], Loss: 0.0164

train-loss: 0.2709, train-acc: 99.8431
validation loss: 0.6181, validation acc: 83.9506

12
Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0118
Epoch [200/200], Step [20/60], Loss: 0.0037
Epoch [200/200], Step [40/60], Loss: 0.0024

train-loss: 0.0372, train-acc: 100.0000
validation loss: 0.4207, validation acc: 87.5772

13
Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0007
Epoch [200/200], Step [20/60], Loss: 0.0003
Epoch [200/200], Step [40/60], Loss: 0.0008

train-loss: 0.0106, train-acc: 100.0000
validation loss: 0.3718, validation acc: 89.5062

14

Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0007
Epoch [200/200], Step [20/60], Loss: 0.0007
Epoch [200/200], Step [40/60], Loss: 0.0011

train-loss: 0.0944, train-acc: 99.8954
validation loss: 1.1262, validation acc: 77.5463

15
Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0010
Epoch [200/200], Step [20/60], Loss: 0.0030
Epoch [200/200], Step [40/60], Loss: 0.0056

train-loss: 0.0941, train-acc: 100.0000
validation loss: 1.0634, validation acc: 80.4784

16

Epoch [200/200], Step [0/60], Loss: 0.0012
Epoch [200/200], Step [20/60], Loss: 0.0009
Epoch [200/200], Step [40/60], Loss: 0.0003

train-loss: 0.0133, train-acc: 100.0000
validation loss: 0.3253, validation acc: 91.7438

17
Epoch 200

Epoch [200/200], Step [0/60], Loss: 0.0009
Epoch [200/200], Step [20/60], Loss: 0.0005
Epoch [200/200], Step [40/60], Loss: 0.0008

train-loss: 0.0345, train-acc: 100.0000
validation loss: 0.5969, validation acc: 90.1235

18. 
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0003
Epoch [200/200], Step [20/65], Loss: 0.0010
Epoch [200/200], Step [40/65], Loss: 0.0006
Epoch [200/200], Step [60/65], Loss: 0.0006

train-loss: 0.0147, train-acc: 100.0000
validation loss: 0.2300, validation acc: 94.7293

19.
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0002
Epoch [200/200], Step [20/65], Loss: 0.0002
Epoch [200/200], Step [40/65], Loss: 0.0002
Epoch [200/200], Step [60/65], Loss: 0.0007

train-loss: 0.0849, train-acc: 100.0000
validation loss: 0.9336, validation acc: 82.9772

20.
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0003
Epoch [200/200], Step [20/65], Loss: 0.0003
Epoch [200/200], Step [40/65], Loss: 0.0002
Epoch [200/200], Step [60/65], Loss: 0.0003

train-loss: 0.0315, train-acc: 100.0000
validation loss: 0.4036, validation acc: 92.6638

21.
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0007
Epoch [200/200], Step [20/65], Loss: 0.0003
Epoch [200/200], Step [40/65], Loss: 0.0002
Epoch [200/200], Step [60/65], Loss: 0.0003

train-loss: 0.0105, train-acc: 100.0000
validation loss: 0.2382, validation acc: 93.8746

22. stable
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0024
Epoch [200/200], Step [20/65], Loss: 0.0019
Epoch [200/200], Step [40/65], Loss: 0.0025
Epoch [200/200], Step [60/65], Loss: 0.0013

train-loss: 0.0372, train-acc: 100.0000
validation loss: 0.4067, validation acc: 88.3191

23.  stable
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0292
Epoch [200/200], Step [20/65], Loss: 0.0557
Epoch [200/200], Step [40/65], Loss: 0.0236
Epoch [200/200], Step [60/65], Loss: 0.0218

train-loss: 0.2407, train-acc: 99.7592
validation loss: 0.5032, validation acc: 88.8889

24.
Epoch 200

Epoch [200/200], Step [0/33], Loss: 0.0004
Epoch [200/200], Step [20/33], Loss: 0.0006

train-loss: 0.0131, train-acc: 100.0000
validation loss: 0.2981, validation acc: 92.2365

25.
Epoch 200

Epoch [200/200], Step [0/33], Loss: 0.0006
Epoch [200/200], Step [20/33], Loss: 0.0005

train-loss: 0.0120, train-acc: 100.0000
validation loss: 0.3036, validation acc: 93.6610

26.
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0002
Epoch [200/200], Step [20/65], Loss: 0.0006
Epoch [200/200], Step [40/65], Loss: 0.0002
Epoch [200/200], Step [60/65], Loss: 0.0002

train-loss: 0.0085, train-acc: 100.0000
validation loss: 0.1842, validation acc: 94.8006

27. stable
Epoch 200

Epoch [200/200], Step [0/65], Loss: 0.0002
Epoch [200/200], Step [20/65], Loss: 0.0001
Epoch [200/200], Step [40/65], Loss: 0.0001
Epoch [200/200], Step [60/65], Loss: 0.0001

train-loss: 0.0106, train-acc: 100.0000
validation loss: 0.1481, validation acc: 96.6524

