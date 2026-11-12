# LivenessNet layer order (parsed from src/architectures.py)

| position | type | name | mark |
|---|---|---|---|
| 1 | Input | input_layer |  |
| 2 | Conv2D | conv1_1 |  |
| 3 | Activation | relu1_1 |  |
| 4 | BatchNormalization | bn1_1 | BN: recalibrated |
| 5 | Conv2D | conv1_2 |  |
| 6 | Activation | relu1_2 |  |
| 7 | BatchNormalization | bn1_2 | BN: recalibrated |
| 8 | MaxPooling2D | maxpool1 |  |
| 9 | Dropout | dropout1 | dropout: off in inference |
| 10 | Conv2D | conv2_1 |  |
| 11 | Activation | relu2_1 |  |
| 12 | BatchNormalization | bn2_1 | BN: recalibrated |
| 13 | Conv2D | conv2_2 |  |
| 14 | Activation | relu2_2 |  |
| 15 | BatchNormalization | bn2_2 | BN: recalibrated |
| 16 | MaxPooling2D | maxpool2 |  |
| 17 | Dropout | dropout2 | dropout: off in inference |
| 18 | Flatten | flatten |  |
| 19 | Dense | dense1 |  |
| 20 | BatchNormalization | bn_dense | BN: recalibrated |
| 21 | Dropout | dropout_dense | dropout: off in inference |
| 22 | Dense | output |  |
