# CS515-Deep_Learning

## Parameters

### Architecture

| Parameter | Present in the baseline | Description|

| "--hidden_sizes" | "512 256 128" | Hidden layer sizes |

| "--activation" | "relu" | Activation function: "relu", "leakyrelu", "gelu" |

| "--dropout" | "0.3" | Dropout ration |

| "--bn_after_act" | off | Put Batch Normalization after activation (default: before) |

### Training

| Parameter | Present in the baseline | Description|

| "--epochs" | "10" | Number of training epochs |

| "--lr" | "0.001" | Learning rate | 

| "--batch_size" | "64" | Batch size |

| "--regularizer" | "l2" | Regularization type: "l1", "l2" |

| "--scheduler" | "steplr" | LR scheduler: "steplr", "reducelronplateau", "cosineannealiglr" |

| "--early_stop" | "0" | Early stopping patience (0= disabled) |

### Visualization

| Parameter | Present in the baseline | Description|

| "--visualize" | "torchviz", "curves" , "tsne" | Generate plots (multiple plots is possible)


### Example usage

## use GeLU activation function

'python3 main.py --activation gelu'

## use L1 regularization

'python3 main.py --regularizer l1'

## use 20 epochs for training and reducelronplateau learning rate scheduler

'python3 main.py --epochs 20 --scheduler reducelronplateau'

## Custom run

'python3 main.py --epochs 20 --lr 0.0005 --activation gelu --dropout 0.5 --regularizer l1'
