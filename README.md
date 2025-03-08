# Efficient Ensemble Model for Facial Expression Recognition

## Overview
This repository contains the Ensemble Learning for Facial Expression Recognition (FER) project for the Machine Learning for Data Science (DS503) course at KAIST.
For more details, refer to final_report.pdf and poster.png.


Facial expression recognition (FER) is an essential task in human-computer interaction, customer satisfaction analysis, and accessibility solutions. However, modern deep learning models often require significant computational resources. This project explores **ensemble learning** as an alternative to scaling up deep learning models. We compare **compact models** using ensembling techniques to a large-scale model to determine whether similar accuracy can be achieved with lower computational cost.

## Features
- **Efficient ensemble learning** approach for facial expression recognition.
- **Uses compact models** (ShuffleNet, MobileNet, SqueezeNet) to optimize computation.
- **Performs comparably to larger models** while reducing computational cost.
- **Explores multiple ensemble strategies** (majority voting, simple averaging, weighted averaging).

## Installation
Download the dataset and trained models:
- **Dataset**: [Download](https://drive.google.com/uc?id=1TOiNfQdH8GVWcNBC9v_YktuwXCbdLuq-)
- **Trained Models**: [Download](https://drive.google.com/uc?id=1DFo9738zbmV1MJ2h5tQzO3BEk-Iulf4i)
- **Google Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/1ssc930UmTXBqlF0JMkIio6IYtDZJ3h0J#scrollTo=HldvGAUHme0g)

To run locally:
```bash
python ensembleFER.ipynb
```

## Methodology
Our approach involves training **compact architectures** using diverse training samples to ensure model diversity.

### **1. Model Architecture**
We selected models based on their efficiency and ability to complement each other in feature extraction. 

Using **Grad-CAM visualizations**, we analyzed the focus areas of different architectures to determine the best candidates for ensemble learning. The final selection included:
![Frame 1](https://github.com/user-attachments/assets/4ece102d-271e-4d0e-a52c-e498fad4318d)


- **Ensemble A (Multi-class classifiers)**: Uses ShuffleNet, MobileNet, and SqueezeNet.
- **Ensemble B (Binary classifiers for difficult expressions)**: Specializes in improving recall for Fear, Disgust, and Sadness.
<img width="1189" alt="figure2" src="https://github.com/user-attachments/assets/c6257f84-4798-41a5-a51f-c872e63a5b55" />

### **2. Aggregation Methods**
- **Majority Voting**: Selects the class predicted by most models.
- **Simple Averaging**: Averages probability scores across models.
- **Weighted Averaging**: Assigns model-specific weights to predictions.

## Experiment
### Dataset
We use the **FER-2013** dataset, which contains 35,887 grayscale images across seven expression categories.

### Training Setup
- **Batch Size**: 48
- **GPU**: NVIDIA L4
- **Epochs**: 15
- **Pretrained Weights Used**

## Results
| Model          | Aggregation      | Accuracy (%) | Training Time | Params (M) | MACs (G) |
|---------------|-----------------|--------------|---------------|------------|----------|
| **Ensemble A** | Majority vote   | 66.56        | 35m 28s       | 10.30      | 1.08     |
|               | Simple average  | **67.01**    | 35m 28s       | 10.30      | 1.08     |
|               | Weighted average | 66.95        | 35m 28s       | 10.30      | 1.08     |
| **Ensemble B** | Majority vote   | 60.99        | 72m 12s       | 19.14      | 1.64     |
|               | Simple average  | 60.30        | 72m 12s       | 19.14      | 1.64     |
|               | Weighted average | 56.03        | 72m 12s       | 19.14      | 1.64     |
| **EfficientNet B3** | -          | **68.62**    | 85m 48s       | 10.71      | 1.93     |
| **Human accuracy**  | -          | 65.00        | -             | -          | -        |

- **Ensemble A performs comparably to EfficientNet B3** but requires **half the training time**.
- **Ensemble B showed lower performance**, likely due to model complexity and overfitting.

## Conclusion
Our results indicate that **compact multi-class ensemble models can achieve high accuracy while reducing computational cost**. Future work can focus on **optimizing ensemble selection** and **exploring adaptive weighting strategies**.

## Contributors
**Thoriq Dimas Ahmad**  
Graduate School of Data Science, KAIST  
Email: [dimasat@kaist.ac.kr](mailto:dimasat@kaist.ac.kr)  

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
