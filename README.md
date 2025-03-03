# Efficient Ensemble Model for Facial Expression Recognition

## Authors

**Aghazadeh Mahsa**  
Graduate School of Data Science, KAIST  
Email: [mahsa_agz@kaist.ac.kr](mailto:mahsa_agz@kaist.ac.kr)  

**Devira Fania Ardelia**  
Graduate School of Data Science, KAIST  
Email: [faniadevira@kaist.ac.kr](mailto:faniadevira@kaist.ac.kr)  

**Fridlund Hampus**  
Kim Jaechul Graduate School of AI, KAIST  
Email: [hampusf@kaist.ac.kr](mailto:hampusf@kaist.ac.kr)  

**Natsagdorj Zuv-Uilst**  
Graduate School of Data Science, KAIST  
Email: [zuvuilst@kaist.ac.kr](mailto:zuvuilst@kaist.ac.kr)  

**Thoriq Dimas Ahmad**  
Graduate School of Data Science, KAIST  
Email: [dimasat@kaist.ac.kr](mailto:dimasat@kaist.ac.kr)  

---

## Abstract
Facial expression recognition (FER) models are evolving towards high accuracy through larger architectures, which come at a computational cost. This study investigates the efficiency and accuracy of ensemble methods using compact models compared to a large-scale model. Our experiments show that ensembling three smaller models with simple averaging achieved nearly the same accuracy as the larger model, with significantly lower training time. These findings highlight the potential of ensemble learning for FER with reduced computational overhead.

## Keywords
Facial expression recognition, FER-2013, Ensemble deep learning, Compact models ensembling, Efficient architectures, One-vs-rest binary classifiers

## Introduction
Facial expression recognition (FER) has various applications, including customer service, accessibility for disabled individuals, and human-computer interaction. However, the increasing complexity of deep learning models necessitates greater computational resources. This work explores an alternative approach by leveraging ensemble learning to achieve high performance with lower computational costs.

## Related Works
Deep learning models such as CNNs have significantly improved FER accuracy. Studies have demonstrated that ensemble techniques combining multiple small models can achieve comparable accuracy to larger models while reducing computational demand. We build on this research by proposing an ensemble of compact models optimized for FER.

## Proposed Method: Ensemble Deep Learning for FER
We propose two ensemble models:
<img width="1160" alt="image" src="https://github.com/user-attachments/assets/aa7c472f-ccc7-46b9-9540-4c0adb95ed71" />

- **Ensemble A**: Three multi-class classifiers (ShuffleNet, MobileNet, and SqueezeNet) trained with different data samples.
- **Ensemble B**: A hybrid approach using a multi-class classifier and three binary classifiers focusing on emotions with lower recall (fear, sadness, and disgust).

### Aggregation Methods


Each model in the ensemble predicts a probability for each facial expression class, with different weighting factors assigned to account for model strengths. The total number of networks in the ensemble is denoted by $B$, and predictions are combined using different aggregation methods. The indicator function $I(.)$ ensures that relevant conditions are met for classification. The predicted label from each model is represented as $y_b$, while $f_{b,c}$ indicates the probability assigned to class $c$ by model $b$. The normalized weight $w_{b,c}$ adjusts the final prediction to enhance accuracy by leveraging model-specific performance variations.


#### Majority Voting:
```math
\hat{y} = \arg\max_c \sum_{b=1}^{B} I(y_b = c)
```

#### Simple Averaging:
```math
\hat{y} = \arg\max_c \frac{1}{B} \sum_{b=1}^{B} f_{b,c}
```

#### Weighted Averaging:
```math
\hat{y} = \arg\max_c \sum_{b=1}^{B} w_{b,c} f_{b,c}
```

### Definitions
- $c$ is the facial expression class index.
- $B$ is the total number of networks in the ensemble.
- $I(.)$ is the indicator function, which equals $1$ if the condition inside the function is true and $0$ otherwise.
- $y_b$ is the predicted label from model $b$.
- $f_{b,c}$ is the predicted probability for class $c$ from model $b$.
- $w_{b,c}$ is the normalized weight of prediction probabilities for class $c$ from model $b$.

Each model has different weighting factors, and each class within every model can also have different weighting factors to account for individual model strengths and weaknesses.

## Experiment
### Dataset
We use the **FER-2013** dataset, which contains 35,887 grayscale images across seven expression categories.

### Training Setup
- **Batch Size**: 48
- **GPU**: NVIDIA L4
- **Epochs**: 15
- **Models Initialized with Pre-trained Weights**

### Results
<img width="699" alt="image" src="https://github.com/user-attachments/assets/dd73fc45-c05a-4e06-a220-eff93c9e4de6" />
## Performance Evaluations on FER-2013 Test Set

| Model          | Aggregation      | Accuracy (%) | Training Time | Params (M) | MACs (G) |
|---------------|-----------------|--------------|---------------|------------|----------|
| **Ensemble A** | Majority vote   | 66.56        | 35m 28s       | -          | -        |
|               | Simple average  | **67.01**    | -             | 10.30      | 1.08     |
|               | Weighted average | 66.95        | -             | -          | -        |
| **Ensemble B** | Majority vote   | 60.99        | 72m 12s       | -          | -        |
|               | Simple average  | 60.30        | -             | 19.14      | 1.64     |
|               | Weighted average | 56.03        | -             | -          | -        |
| **EfficientNet B3** | -          | **68.62**    | 85m 48s       | 10.71      | 1.93     |
| **Human accuracy**  | -          | 65.00        | -             | -          | -        |

Our results show that Ensemble A closely matches the accuracy of EfficientNet B3 while training twice as fast.

## Conclusion
Ensemble A, utilizing multi-class classifiers, provides a viable alternative to large FER models with minimal computational trade-offs. Future research may explore optimizing binary classifiers and alternative aggregation methods.

## References
A complete reference list is available in the full paper.

## Code and Data Availability
All code and datasets are available in this repository. For more details, check our [GitHub Repository](https://github.com/yourusername/facial-expression-ensemble).

## Contact
For inquiries, please reach out to **Aghazadeh Mahsa** at [mahsa_agz@kaist.ac.kr](mailto:mahsa_agz@kaist.ac.kr).
