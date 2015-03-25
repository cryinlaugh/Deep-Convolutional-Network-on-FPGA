# Deep-Convolutional-Network-on-FPGA
a simplified version of LeNet5

## Double Precision
|Name     |Vector Size|Vector Size Decoupled Interface|util for test|sim test|build hw|hw test|
|:--------|:---------:|:-----------------------------:|:-----------:|:------:|:------:|:-----:|
|CNN_FW_Conv_V0   |N/A|y|y|y|y|y|
|CNN_FW_Conv_V1   |4,Failed;4,50MHz,SUC;**3,SUC**|y|y|y|y|y|
|CNN_BP_Conv_V0   |N/A|y|y|y|y|y|
|CNN_BP_Conv_V1   |2Failed|y|y|y| | |
|CNN_FW_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_FW_MaxPool_V1|12,Failed;8,ing|y|y|y|y|y|
|CNN_BP_MaxPool_V0|N/A|y|y|y|y|y|
|CNN_BP_MaxPool_V1|12,SUC|y|y|y|y|y|
|CNN_FW_Softmax_V0|Discarded.1| | | | | |
|CNN_FW_Softmax_V1|12,SUC|y|y|y|y|y|
|CNN_BP_Softmax_V0|Discarded.1| | | | | |
|CNN_BP_Softmax_V1|12,ing|y|y|y| | |

