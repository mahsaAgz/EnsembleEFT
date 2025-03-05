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
## Introduction
Recent studies have developed implementations of deep learning models for facial expression recognition (FER) tasks because of their potential application in evaluating customer service satisfaction, human-computer interaction systems, and the development of accessibility features for disability-inclusive systems [6]. 

The current SOTA for image classification and FER revolves around upscaling the model size at the cost of computational resources [1], [7], [8], [10]. The challenge lies in developing models that can accurately identify emotions while maintaining computational efficiency, particularly for real-time applications.

As computational power is neither easy nor affordable to lay hands on, we aim to compare a large model’s accuracy and efficiency to those of smaller models that utilize ensembling techniques. Our vision is to find out if a way exists to achieve comparably great accuracy to that of the larger model, hence achieving great accuracy by using less computational power and without scaling the models. 

We test various ensemble methods, including majority voting, simple averaging and weighted averaging to evaluate the performance in FER tasks. The results show that a all multi-class ensemble of compact models can achieve high accuracy comparable to human performance, with significantly reduced computational demands.
This research aims to provide insights into the trade-offs between accuracy and efficiency in FER, offering practical solutions for real-world applications where computational resources are limited.

## Methodology
***Ensemble deep learning for facial expression recognition.*** The training process is done using different bootstrap samples to ensure that each network learns different features from the FER 2013 dataset. To further increase the diversity of learned features [2], each network is selected from a variety of unique compact architecture families—ShuffleNet[11], MobileNet[4], and SqueezeNet[5]—that have different approaches for extracting features efficiently from the image, as shown in Figure 3.
<img width="1189" alt="figure3" src="https://github.com/user-attachments/assets/a9150554-b5c5-4856-8287-d436efa12f62" />

Furthermore, we explored using one vs rest binary classifiers as the ensemble member candidate, to encourage complementary interaction between the ensemble member. This way, some member could focus on learning difficult facial expressions and others can focus on other expressions.

Individual predictions from each model are then combined using some aggregation function, namely majority voting, Simple averaging and weight averaging, as shown in Figure 1 and 2. Additionally, we use EfficientNet [10] B3 as the larger model to compare with our ensemble method.

We explored 2 ensemble model with different members:
Ensemble A (All multi-class classifiers): ShuffleNet, MobileNet, and SqueezeNet
Ensemble B (With binary classifiers): MobileNet (multi), ShuffleNet (Fear), ShuffleNet (Disgust), and MobileNet (Sad)
<img width="1189" alt="figure2" src="https://github.com/user-attachments/assets/c6257f84-4798-41a5-a51f-c872e63a5b55" />

Ensemble B clarification: We chose to predict disgust, fear, and anger separately due to their consistently low recall in individual models. For each expression, we trained the architecture with the highest recall as a binary classifier, which improved prediction accuracy. For instance, as shown in Figure 4, ShuffleNet correctly identified fear by focusing on the eyes, while MobileNet misclassified it by focusing on the mouth and nose. This approach enhanced the performance of our emotion recognition system.
<img width="1289" alt="figure4" src="https://github.com/user-attachments/assets/1766bbd7-a2b7-45ef-ab0f-d04ce08f46f7" />

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
## Performance Evaluations on FER-2013 Test Set

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


We compared the performance of **Ensemble A**, **Ensemble B**, and **EfficientNet B3** using accuracy, training time, number of parameters, and MACs. Additionally, we evaluated their performance relative to human accuracy.

- **Ensemble A had the highest performance among the ensemble models**, reaching **67.01% accuracy** with simple averaging. Majority voting and weighted averaging resulted in **66.56% and 66.95% accuracy**, respectively.
- **Ensemble B showed lower performance**, with **60.99% accuracy** using majority voting, **60.30% with simple averaging**, and **56.03% with weighted averaging**.
- **EfficientNet B3 achieved the highest accuracy (68.62%)** but required the longest training time.
- **Training time for Ensemble A was 35 minutes and 28 seconds**, which was **faster than Ensemble B (72 minutes) and EfficientNet B3 (85 minutes)**.
- **Human accuracy was 65%**, and both Ensemble A and EfficientNet B3 exceeded this value.

Since the accuracy for **Fear, Disgust, and Sadness** was lower across models, Ensemble B incorporated **binary classifiers** to improve detection. However, its accuracy decreased. Possible reasons include **overconfidence in detecting specific emotions**, leading to **reduced generalization**, and **higher model complexity**, with **more parameters (19.14M vs. 10.30M) and higher MACs (1.64G vs. 1.08G)**, which may have resulted in **overfitting**.

## Discussion
- This finding emphasizes the practicality and efficiency of the all multi-class ensemble approach, particularly given the context of its performance relative to human-level and large-scale model’s accuracy.

- The specialized ensemble approach like Ensemble B didn’t perform as well because it used a more complex approach which resulted in overconfidence (higher recall but lower precision) and more computational cost. 

- It is important to select combinations that can complement each other by considering the diversity of the models rather than increasing the complexity of the ensemble, as supported by the results from Ensemble A with weighted averaging.

- Although weighted averaging was less accurate than simple averaging, It demonstrated that highlighting the strengths of each model can enhance the performance of the model.

## Conclusion
We proposed 2 types of ensemble networks, ensemble A utilizing multi-class ensemble only and ensemble B utilizing one-vs-rest binary classifiers. Ensemble A has shown comparable performance to that of a larger model, while using less computational cost with less than half of the training time of the large model. 

These findings suggest that it is possible to create a high-performance classifier for facial expression image recognition with lower computational costs using ensemble technics as an alternative to scaling. For further research, one could explore optimizing ensemble methods to achieve even greater tradeoff between efficiency and accuracy on facial emotion image recognition tasks. 

## References
1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv:2010.11929, 2020.
2. Ganaie, Mudasir A., et al. "Ensemble deep learning: A review." Engineering Applications of Artificial Intelligence 115 (2022): 105151.
3. Goodfellow, Ian J., et al. "Challenges in representation learning: A report on three machine learning contests." In ICONIP. Proceedings, Part III 20. Springer berlin heidelberg, 2013.
4. Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv:1704.04861, 2017.
5. Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv:1602.07360, 2016.
6. Khaireddin, Yousif, and Zhuofa Chen. "Facial emotion recognition: State of the art performance on FER2013." arXiv:2105.03588, 2021.
7. Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF ICCV. 2021.
8. Pham, Luan, The Huynh Vu, and Tuan Anh Tran. "Facial expression recognition using residual masking network." In ICPR. IEEE, 2021.
9. Shorten, Connor, and Taghi M. Khoshgoftaar. "A survey on image data augmentation for deep learning." Journal of big data 6.1 (2019): 1-48.
10. Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." In ICML. PMLR, 2019.
11. Zhang, Xiangyu, et al. "Shufflenet: An extremely efficient convolutional neural network for mobile devices." Proceedings of the IEEE CVPR. arxiv:1707.01083, 2018.


## Code and Data Availability

