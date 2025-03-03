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
To combine model predictions, we experiment with:
1. **Majority Voting** (Eq. 1)
2. **Simple Averaging** (Eq. 2)
4. **Weighted Averaging** (Eq 3)
<img width="1160" alt="Formula" src="https://github.com/user-attachments/assets/4fe2362a-943e-44a8-b582-9c3cabe000af" />


## Experiment
### Dataset
We use the **FER-2013** dataset, which contains 35,887 grayscale images across seven expression categories.

### Training Setup
- **Batch Size**: 48
- **GPU**: NVIDIA L4
- **Epochs**: 15
- **Models Initialized with Pre-trained Weights**

### Results
| Model | Aggregation | Accuracy (%) | Training Time | Parameters (M) | MACs (G) |
|--------|-------------|-------------|----------------|---------------|----------|
| **Ensemble A** | Simple Average | **67.01** | **35m 28s** | 10.30 | 1.08 |
| **Ensemble A** | Majority Vote | 66.56 | 35m 28s | - | - |
| **Ensemble B** | Simple Average | 60.30 | 72m 12s | 19.14 | 1.64 |
| **EfficientNet B3** | - | 68.62 | 85m 48s | 10.71 | 1.93 |
| **Human Accuracy** | - | 65.00 | - | - | - |

Our results show that Ensemble A closely matches the accuracy of EfficientNet B3 while training twice as fast.

## Conclusion
Ensemble A, utilizing multi-class classifiers, provides a viable alternative to large FER models with minimal computational trade-offs. Future research may explore optimizing binary classifiers and alternative aggregation methods.

## References
A complete reference list is available in the full paper.

## Code and Data Availability
All code and datasets are available in this repository. For more details, check our [GitHub Repository](https://github.com/yourusername/facial-expression-ensemble).

## Contact
For inquiries, please reach out to **Aghazadeh Mahsa** at [mahsa_agz@kaist.ac.kr](mailto:mahsa_agz@kaist.ac.kr).
