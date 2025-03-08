# Efficient Ensemble Model for Facial Expression Recognition


## Authors



## Introduction
Facial expression recognition (FER) is an essential task in human-computer interaction, customer satisfaction analysis, and accessibility solutions. However, modern deep learning models often require significant computational resources. This project explores **ensemble learning** as an alternative to scaling up deep learning models. We compare **compact models** using ensembling techniques to a large-scale model to determine whether similar accuracy can be achieved with lower computational cost.

## Methodology
### Ensemble deep learning for facial expression recognition.
The training process is done using different bootstrap samples to ensure that each network learns different features from the FER 2013 dataset. To further increase the diversity of learned features, each network is selected from a variety of unique compact architecture families—ShuffleNet, MobileNet, and SqueezeNet, that have different approaches for extracting features efficiently from the image, as shown in Figure 3.
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

Aggregation methods used:
- **Majority Voting**: Selects the class predicted by most models. $\hat{y} = \arg\max_c \sum_{b=1}^{B} I(y_b = c)$
- **Simple Averaging**: Averages probability scores across models. $\hat{y} = \arg\max_c \frac{1}{B} \sum_{b=1}^{B} f_{b,c}$
- **Weighted Averaging**: Assigns model-specific weights to predictions. $\hat{y} = \arg\max_c \sum_{b=1}^{B} w_{b,c} f_{b,c}$

  where
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

## Installation

The dataset and trained models used in this study are available for public access. 

- **Dataset**: [here](https://drive.google.com/uc?id=1TOiNfQdH8GVWcNBC9v_YktuwXCbdLuq-) 
- **Trained Models**: [here](https://drive.google.com/uc?id=1DFo9738zbmV1MJ2h5tQzO3BEk-Iulf4i)
- **Google Colab Notebook**: The experiment can be replicated in a Colab environment for optimal performance. The notebook can be accessed here: [link](https://colab.research.google.com/drive/1ssc930UmTXBqlF0JMkIio6IYtDZJ3h0J#scrollTo=HldvGAUHme0g)

To run the experiment locally, download the `ensembleFER.ipynb` notebook and execute it in a compatible environment. 
