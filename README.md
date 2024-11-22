# Automating Thematic-Analysis

**Here's a little summary, if you want to know more details then simply keep on scrolling and read below**

Automating Insight Reports with BERT and BERTopic

This project demonstrates automating the generation of insightful reports by leveraging BERT for text classification and BERTopic for topic modeling. It involves three key components:

- Text Classification: Employing BERT to categorize student feedback efficiently.
- Topic Modeling: Using BERTopic to uncover themes and visualize data.
- Report Generation: Merging insights from both models to create detailed, visually engaging reports, saving time and resources.

For queries, contact: malikrahil77@gmail.com.


# Detailed Report


One of the fundamental techniques in qualitative research is thematic analysis, which is used to find, examine systematically, and document patterns in data. It is also useful in exploratory research where understanding the underlying themes is essential. This flexible method allows researchers to adapt it to various research questions and data types. Virginia Braun and Victoria Clarke formalized a flexible approach to theme identification, it requires researchers to move back and forth across the data, codes, and themes throughout the analysis, making it an inherently iterative method. 

I automate the process of thematic analysis, Using BERT for Text classification and BERTopic for Topic Modelling. Then, I generated a report that showcases the information in a concise format.

## Why? 
Why did I try to automate the process of manual thematic analysis?

We can point out that managing large datasets can become cumbersome due to the repetitive coding and review cycles. Manual thematic analysis is also prone to subjective bias, where the researcher's interpretation can influence the data coding process. The time-consuming nature of the process is easily highlighted particularly in the context of analysing vast datasets. The manual approach takes a lot of time and effort from the researchers to code the data and find themes, and when the size of the data increases exponentially this method becomes infeasible. Even though this method is thorough, it takes a long time and is vulnerable to researcher bias because it depends so much on the individual interpretation of the data. Even though thematic analysis is a comprehensive procedure by hand, there are a lot of scalability and consistency issues, especially with huge datasets. Given the challenges outlined above, the need for automation in thematic analysis becomes clear. Automation offers a solution to the time constraints and potential biases inherent in manual analysis.

This project demonstrates how combining the help of large language models(LLMs) can enhance the efficiency of qualitative analysis by reducing the manual labour required while still providing insightful results that align with traditional analysis methods.

Note: The data this project works on was National Student Survey data specific to Brunel University, the data is confidential and access is provided to only people at Brunel.


## Topic Modelling using BERTopic 

The objective is to develop and implement a topic modeling using BERTopic to automatically identify and categorize themes within large textual datasets, demonstrating its application for extracting insights from survey responses.

The National Student Survey (NSS) dataset used for topic modeling includes survey responses spanning three years (2021–2023) from students at Brunel University. These datasets were rich in textual data, excellent for our purpose of topic modeling which is an unsupervised learning technique. Each row in the dataset includes both a positive and negative comment, with both comments coming from the same individual.

### Data Pre-processing

The preprocessing for topic modeling involved several steps aimed at cleaning and preparing the text data for the BERTopic model. This involved the following processes: 
- Text cleaning: Removal of unnecessary elements such as numbers, special characters, punctuations, and extra whitespaces. This ensured that the model focused only on the meaningful content of the text. 
- Stopword removal: Standard English stopwords, such as "and," "the," "is," etc., were removed using CountVectorizer to reduce noise in the textual data. 
- Lemmatization: The process of lemmatizing the words was implemented using the NLTK library, ensuring that words were reduced to their base forms. For instance, "running," "ran," and "runs" were reduced to "run." This ensured that different forms of the same word were treated as a single entity, enhancing the quality of the topics generated. 
- Duplicate removal: Any duplicate entries were eliminated to avoid redundancy in topic identification. 
- Missing rows removal: any rows with missing values in the positive or negative comment fields were removed. The proportion of such rows was small, and since the models require complete data with no missing values, these rows were excluded from further analysis. 

### Model Implementation

BERTopic leverages pre-trained transformers to create semantic embeddings, which are then clustered using algorithms like UMAP and HDBSCAN. This knowledge allows us to fine-tune several parameters in key components, such as UMAP and HDBSCAN, to improve our topic modeling. There are a few hyperparameters that we can adjust to optimize the model’s performance, with the embedding model being one of the most significant.

#### Embedding Model

To generate high-quality word embeddings, we used various sentence transformer models 
such as: 
- all-distilroberta-v1 with max sequence length 512 and dimension 768
- all-mpnet-base-v2 with max sequence length 384 and dimension 768 
- all-MiniLM-L12-v2 with max sequence length 256 and dimension 384 
- all-MiniLM-L6-v2 with max sequence length 256 and dimension 384 
- We also experiment using 'roberta-base' for document embedding.

#### Dimensionality Reduction Method

UMAP(Uniform Manifold Approximation and Projection) was used to reduce the high dimensional embeddings into a lower-dimensional space that could be efficiently clustered. It plays a key role in improving the visual clarity of the topics as well as aiding the clustering process. Several UMAP parameters were fine-tuned to optimize performance: 
- n_neighbors: This parameter defines how many nearby points UMAP considers when 
constructing the underlying manifold. A higher value tends to capture broader global 
structures, whereas we picked a lower value of 10 which focuses on local 
relationships between data points,which is more suitable for identifying finer-grained 
topics.
- min_dist: This parameter controls how closely UMAP can pack points together in the 
low-dimensional space. We pick a smaller value so that it preserves more of the local 
detail by allowing points to be closer together, while a larger value spreads them out 
more evenly. 
- n_components: This represents the number of dimensions into which UMAP reduces 
the data. Set to 5 to reduce the data to a manageable dimensionality while retaining 
relevant information.

#### Clustering Method

HDBSCAN was employed to cluster the reduced embeddings into meaningful topics. It works by identifying dense clusters of data points, which it treats as topics, while filtering out noise or outliers. It also provides flexibility with several key hyperparameters: 
- min_cluster_size: This value sets the minimum size a group of data points must have to be considered a valid cluster. Smaller values allow for more granular topics, while larger values ensure the model forms more substantial clusters. But when we aimed for larger values, all of them collapsed into one topic. A value of 10 was chosen to ensure that clusters (topics) were meaningful but not overly large. 
- min_samples: This controls how conservative the clustering algorithm is by specifying the minimum number of points needed to form a cluster. Higher values ensure more reliable clusters but can cause smaller clusters to be ignored. Set to 5, balancing the need for reliable clusters with the relatively small size of the dataset.  
- metric: This defines the distance metric that HDBSCAN uses to measure similarity between data points. Common metrics include Euclidean distance or cosine similarity, with each having different impacts on the clustering results. Euclidean distance was selected based on its superior performance with smaller datasets like ours.
  
Adjusting these parameters greatly influenced the performance and quality of the topics generated by BERTopic, allowing us to customize the model to better fit our dataset and research goals.

#### Evaluation Metrics

The topics were visualized using an Intertopic Distance Map, which allowed us to examine the relationships between different topics, and a bar chart of topic word scores, providing insights into the most important words per topic. By reviewing the topics, we can gauge how coherent they are with their representative documents. Although evaluating topic generation is a bit subjective, numerical metrics like Topic Coherence and Topic Diversity can provide more objective measures for comparing different models and hyperparameter configurations. 

Topic Coherence measures the semantic similarity between the words in a topic. A high coherence score suggests that the words in a topic are more related to each other, indicating better topic quality. 

Topic Diversity quantifies the uniqueness of words across topics. A higher diversity score indicates that the topics have distinct sets of words, minimizing overlap between topics. This balance between coherence and diversity helps us achieve meaningful and distinct topics.  

### Results

To evaluate the performance of the topic modeling, we analyzed the top topics generated by the model. Figure 3.1 showcases the most significant topics identified. The topic modeling results revealed several coherent topics, let us look at a few: 
- Topic 0: Associated with terms such as "student," "module," "lecturer," and "course," representing common academic themes. 
- Topic 1: Contained terms like "covid," "pandemic," and "online," indicating discussions around the COVID-19 pandemic and its impact on their education. 
- Topic 7: Included words such as "email," "reply," "respond," and "quickly," with representative documents discussing the promptness of email responses from staff and supervisors.

<img width="473" alt="image" src="https://github.com/user-attachments/assets/082b5f1d-4863-477f-82dd-4c5258c4368e">

We can further validate these topics by reviewing their representative documents. For example, let us consider Topic 7:

Topic words for topic 7: 

('email', 0.32), ('reply', 0.19), ('respond', 0.12), ('replying', 0.09), ('quickly', 0.08), ('staff', 0.08), ('time', 0.07), ('quick', 0.05), ('lecturer', 0.05), ('long', 0.05) 

Representative documents for topic 7: 
- 'Staff always helpful quick reply email' 
- 'lecturer respond email sometimes email lot' 
- 'staff helpful always reply back email'

These words and documents collectively indicate that Topic 7 refers to communication with staff and lecturers, specifically the helpfulness and speed of email responses. The alignment between topic words and representative documents demonstrates the model's ability to capture meaningful themes within the dataset. 

To further assess the model's performance, we relied on Topic Coherence and Topic Diversity. While fine-tuning the model's hyperparameters, we experimented with several sentence transformers: 

<img width="419" alt="image" src="https://github.com/user-attachments/assets/09ec5949-56e5-424c-942e-93a081af959a">

Performance-wise, we find that all-distilroberta-v1 provides the best balance between Topic Coherence and Topic Diversity, making it the most suitable transformer for our analysis. Upon subjective review, this model also produced the most coherent and meaningful topic words. 

In terms of training time, roberta-base took over one minute to train, while the other models due to their smaller architectures finished training within 20 seconds. This reduced training 
time is likely due to the relatively small dataset we used.

Despite the efforts, we could not achieve a higher coherence score. However, the topics and topic words generated still provide valuable insights and align well with the underlying documents. While coherence scores serve as an important metric, the subjective evaluation of the topics indicates that the results are still meaningful and offer relevant information for further analysis. 


In the end, we export the topics, their frequency, the top words and supporting documents using csv and json. The information is presented in the report that we generate. 


## Text Classification using BERT 

The goal here is to create a text classification model using BERT to accurately predict categories from raw text data, with an emphasis on improving accuracy and reducing manual labour in text annotation tasks. 

### Dataset description

The 2021 dataset is the most detailed and structured. It contains approximately 1,020 rows, where each row represents a unique response from a student. The dataset includes the following key elements: 
- Textual Data: Each row contains two textual fields, a Positive Comment and a Negative Comment, both written by the same individual. These comments describe the student’s experience with various aspects of the university. 
- Categorical Flags: In addition to the textual data, the 2021 dataset contains 11 flag columns, where each flag corresponds to a specific topic. These topics represent key areas of student feedback, such as: 
  - Overall  
  - Teaching 
  - Covid-19 Pandemic  
  - Learning Opportunities 
  - Assessment and Feedback 
  - Academic Staff and Support 
  - Organization and Management 
  - Learning Resources 
  - Learning Community 
  - Student Voice 
  - Student Union and Related Services


For each student comment, the corresponding flag columns indicate whether the comment discusses these topics, with a value of 1 for relevant topics and 0 for irrelevant ones. Since a single comment can address multiple topics, a row may have multiple flags set to 1, reflecting the multi-label nature of the task.

The 2022 and 2023 datasets are less detailed than the 2021 dataset. They contain only the textual data without the categorical flags. These datasets also consist of approximately 1,000 to 1,090 rows per year. Each row includes a Positive Comment and a Negative Comment, similar to the 2021 dataset. However, these datasets do not provide pre-labeled topic categories, making them suitable for the prediction task, where the model trained on the 2021 data is used to infer the relevant categories for each comment. This way we will be able to generate new information for our report. 

Exploratory data analysis revealed slight imbalances in the dataset. Some categories are more frequently flagged than others. To address this imbalance, resampling techniques were employed to balance the dataset, ensuring the model did not overfit the majority classes while still learning effectively from the minority classes.

### Data Pre-processing

For the pre-processing of text classification, the relevant columns were selected, including the 'Positive comment', 'Negative comment', and all the flag columns representing categories. Each of these columns was prefixed with "Flag:", such as 'Flag: Overall', 'Flag: Teaching', and so forth, covering categories like 'Overall', 'Covid-19 pandemic', 'Teaching', 'Learning Opportunities', 'Assessment and Feedback', 'Academic staff and support', 'Organisation and management', 'Learning resources', 'Learning community', 'Student voice', and 'Student Union and related'. The prefix "Flag:" was removed from each column name to simplify the dataset. 

Since the flags assigned to a row referred to both the positive and negative comments, these comments were combined and separated by a period. This merged text formed the only comment column that contained the textual data moving forward. 

Exploratory analysis revealed that the data was slightly imbalanced. As this was a multi-label problem, where multiple categories could apply to each comment, the decision was made not to balance each category individually. Instead, a strategy of slight down-sampling of the majority classes and up-sampling of the minority classes was applied. This ensured that the model did not overfit the majority classes while still learning from the minority classes. To 
achieve this, every combination of labels was examined. For instance, the combination (0,1,1,0,0,0,0,0,0,0,0) represents flags for Covid-19 and Teaching. The number of rows for each specific combination was counted, and the data was resampled to balance these combinations. More samples were taken from combinations with fewer instances taken from combinations that were overrepresented. This balancing method had a significant impact on the model's accuracy, which will be discussed in detail later. 

### Model Implementation

To ensure the model was trained effectively while preserving data for testing, a train-test validation split of 70:15:15 was applied. 

#### Tokenizer

The BERT Tokenizer (BertTokenizer) was initialized, using the pre-trained model 'bert-base-uncased'. This version of BERT was chosen because it balances computational efficiency and performance. Although BERT-large offers greater capacity, it was not necessary for the scope of this task, and the hardware available did not meet the high computational requirements for running the larger model. The tokenizer was employed to convert chunks of text into BERT
compatible encodings. Similarly, ‘distilbert/distilbert-base-uncased’ was also tested. Distilbert is smaller, faster and lighter version of BERT. Even though it retains 97% of its language understanding capabilities, we pick regular bert as our main/base model.

PyTorch was used to build the added layers for fine-tuning BERT for our specific task. Custom classes, such as a custom dataset and data loader functions, were created to handle the data. A separate class was defined to construct the layers that would be added on top of BERT. A linear layer with 768 units was added, followed by an output layer containing 11 units, corresponding to the 11 categories in the classification task. To mitigate overfitting, a dropout of 0.5 was applied to the linear layer. The issue of overfitting will be addressed in detail later. 

#### Loss Function

For the loss function, **Binary Cross Entropy with Logits Loss** was chosen. This function combines a sigmoid activation layer with binary cross-entropy in a single step. The sigmoid layer maps the output values between 0 and 1, making it suitable for multi-label classification problems where each label is predicted independently. The binary cross-entropy component then calculates the loss by comparing the predicted probabilities against the true labels, which is particularly useful for handling multi-label classification tasks like ours, where each instance may belong to multiple categories. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability. 

#### Optimizer

For the optimizer, several options were tested, including AdamW, AdaFactor, and Adagrad. Among these, **AdamW** provided the best performance for this task. AdamW is an improved version of the Adam optimizer that corrects the weight decay implementation, leading to better generalization by preventing overfitting during training. It modifies the Adam algorithm by decoupling the weight decay from the gradient update, which proved effective in this case.

For model evaluation, the model's accuracy was monitored at each epoch. If the accuracy improved compared to the previous epoch, the model's weights and relevant information were saved. The accuracy and loss history for each epoch were also recorded. 

#### Evaluation Metrics

Evaluation metrics such as accuracy and loss are crucial for assessing model performance. Accuracy provides a straightforward measure of how often the model's predictions match the true labels, offering a high-level view of overall performance. The loss, on the other hand, measures the difference between the predicted values and the actual values, helping to understand how well the model is learning. Monitoring these metrics across epochs is essential for diagnosing issues such as overfitting or underfitting.

The accuracy vs. epoch and loss vs epoch graph is particularly useful for visualizing how the model’s performance improves over time. A steady increase in accuracy and a decrease in loss typically indicate successful learning and convergence. Conversely, if the accuracy plateaus or the loss fails to decrease, it may signal the need for further hyperparameter tuning or model adjustments. 

In addition to accuracy and loss, various classification metrics are employed to evaluate model performance comprehensively, especially in multi-label classification tasks. These include: 
- Precision: The proportion of true positive predictions among all positive predictions made by the model. 
- Recall: The proportion of true positive predictions among all actual positives in the dataset.
- F1 Score: The harmonic mean of precision and recall, providing a balance between the two metrics.
  
These metrics offered insights into different aspects of the model's performance, such as its ability to identify relevant instances (recall) and the accuracy of its positive predictions (precision). Utilizing a combination of these metrics provides a more nuanced understanding of the model's effectiveness in a multi-label classification context.


### Results

The testing accuracy, which is evaluated on data unseen by the model, reached 98.27%. This accuracy was significantly influenced by the application of up-sampling and down-sampling techniques, which addressed class imbalance issues within the dataset. Prior to applying these techniques, the model's accuracy did not exceed 80%, indicating the critical role of sampling in improving model performance.  

The model’s training time varied based on the hyperparameters and optimizers used. The final model, which achieved the best performance in the shortest time, required 26 minutes for training. This efficiency was partly due to using a maximum sequence length of 256 tokens, which optimized both training time and resource utilization. Increasing the sequence length to 512 tokens led to excessive training times (over 200 minutes) and overfitting issues, with training accuracy reaching 99% while testing and validation accuracy dropped to 87%. The max length of 256 was more than enough for the model to learn and correctly predict labels. 

The loss versus epoch graph in fig 4.1 reveals that training and validation losses intersected around the 4th epoch. Beyond this point, the model began to overfit, although this overfitting was relatively minor. The careful adjustment of hyperparameters, particularly the learning rate, was crucial for optimizing performance. A learning rate of 1e-4 proved to be most effective, while higher learning rates caused increased loss and poor model performance.

<img width="426" alt="image" src="https://github.com/user-attachments/assets/460c410f-36cc-4ac4-9498-bf978b481ecc">

While accuracy is an important metric, it does not provide a complete picture of the model's performance. Accuracy only indicates how closely the predicted values match the true labels. To gain more insight, we also used a confusion matrix to evaluate the classification performance across all categories. 

<img width="400" alt="image" src="https://github.com/user-attachments/assets/9f45cc64-35c4-45da-b7f1-f2345bf37bb8">

The confusion matrix in table 4.1 reveals excellent classification performance for almost every category. The F1 scores for many categories are close to 1, which indicates that the model performs well in balancing precision and recall. The absence of false assignments in categories not present in a given row, and the correct assignment of the relevant categories, further supports the model's strong predictive abilities. 

To validate the model's performance on unseen data, we tested it with the sentence: "I can tell that I got good teachers to learn from even though we were struck with COVID-19." The model correctly predicted the labels ‘COVID-19’ and ‘Teaching’, demonstrating its real-world applicability.

In relation to hyperparameter tuning, we found that a learning rate of 1e-04 provided the best results. Higher learning rates led to an increase in loss after a few epochs, resulting in poor model performance. Additionally, we experimented with increasing the maximum length to take advantage of the 512 tokens allowed by the 'bert-base-uncased' model. However, this adjustment significantly extended the training time to over 200 minutes and caused the model 
to overfit. As a result, the testing and validation accuracy dropped to 87%, while the training accuracy soared to 99%. Sticking to 256 was better for training time and efficient use of computational resources too. 

Since only the 2021 data was labelled, I used the trained model to predict labels for the 2022 and 2023 datasets, exporting the results into a separate Excel file. This approach allows for automatic labelling of previously unlabelled data, which could significantly streamline the coding phase of thematic analysis. By automating this process, researchers could reduce the manual effort required to categorize comments, accelerating the analysis workflow.

## Generating the Report

The task is to use the predicted data from Text Classification and themes from topic Modelling then generate a report which uses data visualization and showcases the useful information extracted from the new data generated.

The insights derived from the BERTopic analysis were integrated into the report to enhance its depth and clarity.  

<img width="348" alt="image" src="https://github.com/user-attachments/assets/70db7530-6842-4458-960d-bad914388b89">

From information collected from text classification, we can do more like 

<img width="445" alt="image" src="https://github.com/user-attachments/assets/c03d1a9f-b4fe-4b62-a9ea-237c36bc8ffc">

<img width="407" alt="image" src="https://github.com/user-attachments/assets/3a37bcb7-1df5-483e-b182-e1d13c01d5f7">

 After an overview of the top-mentioned categories, the report delves into each category individually, providing detailed information on their respective counts. Additionally, sample comments are included for each category, 
offering insights into the specific feedback from students and allowing for a clearer understanding of the sentiments and themes expressed in their responses. 

We also export an excel graph separately as can be seen from figure 4.5, here we get the top topics and their percentage in the mix for each sentiment for each year in a visually appealing manner.

<img width="428" alt="image" src="https://github.com/user-attachments/assets/944b0f33-579e-459d-8f39-da14b251c2d1">



This is how we generated a report from the information collected by leveraging large language model BERT and BERTopic. This way we save a lot of man-power and resources by generating report automatically, simply pass the data and do a little bit of tweaking here and there and there you have it.

### **Thanks for reading**

The data that was used in this project is confidential. If you want to re-apply these steps to a different dataset, you will need to tweak the code accordingly but the concept remains same. 

### **Feel free to contact for more information or just have a little chat :)**
malikrahil77@gmail.com














