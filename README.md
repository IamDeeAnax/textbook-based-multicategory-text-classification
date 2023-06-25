# Multi-Class Text Classification with three predefined model

## Introduction
Text classification is one of the exciting parts of NLP. 
Making machines understand natural language and be able to perform tasks like classifying each word or phrase based on the context is one of the challenging tasks in NLP. 

In this task, we focus on multi-class text classification with TensorFlow starting with raw text and using the embedding space. TensorFlow is an open-source machine-learning library with varieties of models that can be used for text classification tasks. 

Three predefined models will be used on raw text data for classification tasks. The three embedding models include Neural Network Language Model with 128 dimensions and normalization (NNLM-128- with-normalization), USE (Universal Sentence Encoder), and BERT (Bidirectional Encoder Representations from Transformers). These embedding models convert text data into numerical representations called vectors.

Before using the embedding models, the raw text will be cleaned by removing punctuations, stop words, new lines, words containing numbers, and text in square brackets, making the text lowercase, and some other preprocessing steps. 

The task aims to test the three pre-trained embedding models on the preprocessed data to generate numerical representation, which will be used to train a classification model and compare the results using different performance metrics. And to have more understanding of TensorFlow text classification. 

![Picture1](https://github.com/IamDeeAnax/NLP_task/assets/111533591/b4e7fbc4-54e8-43f9-abda-461605966cce)


## Usage

1. Clone or download the project repository to your local machine.
   - To download the code, visit the repository's page at https://github.com/IamDeeAnax/NLP_task. Click on the "Code" button and select "Download ZIP".
   - To clone the repository using git, open a terminal and navigate to the desired directory. Run the command "git clone https://github.com/IamDeeAnax/NLP_task" to create a local copy of the repository.

2. Open Jupyter Notebook and navigate to the project directory.

3. Open the .ipynb file containing the project code.
    - You can also click the "Open in colab" at the top of the code.
    
4. Follow the instructions provided in the notebook to run the project and explore its functionalities.

5. You can modify the code and experiment with different inputs to see the results.


## Contributing

Contributions to this project are welcome. If you encounter any issues, have suggestions, or want to contribute improvements, please follow these guidelines:

- Fork the repository and clone it to your local machine.
- Create a new branch for your contributions.
- Make your changes and test them thoroughly.
- Commit your changes and push them to your forked repository.
- Submit a pull request, describing the changes you have made.

Please ensure that your contributions adhere to the project's coding conventions and style guidelines.
