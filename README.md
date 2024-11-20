# Automating Thematic-Analysis

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
- Text cleaning: Removal of unnecessary elements such as numbers, special 
characters, punctuations, and extra whitespaces. This ensured that the model 
focused only on the meaningful content of the text. 
- Stopword removal: Standard English stopwords, such as "and," "the," "is," etc., were 
removed using CountVectorizer to reduce noise in the textual data. 
- Lemmatization: The process of lemmatizing the words was implemented using the 
NLTK library, ensuring that words were reduced to their base forms. For instance, 
"running," "ran," and "runs" were reduced to "run." This ensured that different forms of 
the same word were treated as a single entity, enhancing the quality of the topics 
generated. 
- Duplicate removal: Any duplicate entries were eliminated to avoid redundancy in topic 
identification. 
- Missing rows removal: any rows with missing values in the positive or negative 
comment fields were removed. The proportion of such rows was small, and since the 
models require complete data with no missing values, these rows were excluded from 
further analysis. 

### Model Implementation

BERTopic leverages pre-trained transformers to create semantic embeddings, which are then clustered using algorithms like UMAP and HDBSCAN. This knowledge allows us to fine-tune several parameters in key 
components, such as UMAP and HDBSCAN, to improve our topic modeling. There are a few 
hyperparameters that we can adjust to optimize the model’s performance, with the embedding 
model being one of the most significant.

## Text Classification using BERT 

The goal here is to create a text classification model using BERT to accurately predict categories from raw text data, with an emphasis on improving accuracy and reducing manual labour in text annotation tasks. 

## Generating the Report

The task is to use the predicted data from Text Classification and themes from topic Modelling then generate a report which uses data visualization and showcases the useful information extracted from the new data generated.



