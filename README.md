# Multi-Class Text Classification with three predefined model

## Introduction
Text classification is one of the exciting parts of NLP. Making machines understand natural language and be able to perform tasks like classifying each word or phrase based on the context is one of the challenging tasks in NLP. In this task, we focus on multi-class text classification with TensorFlow starting with raw text and using the embedding space.
 
Three predefined models will be used on raw text data for classification tasks. The three embedding models include Neural Network Language Model with 128 dimensions and normalization (NNLM-128- with-normalization), USE (Universal Sentence Encoder), and BERT (Bidirectional Encoder Representations from Transformers). These embedding models convert text data into numerical representations called vectors.

The task aims to test the three pre-trained embedding models on the preprocessed data to generate numerical representation, which will be used to train a classification model and compare the results using different performance metrics. And to have more understanding of TensorFlow text classification. 

![Picture1](https://github.com/IamDeeAnax/NLP_task/assets/111533591/843dd0c4-1854-404c-8632-81e5455d360a)

## The Application

The machine learning model is served through a web application built with Streamlit. The application takes raw text as input, preprocesses the text, makes a prediction using the machine learning model, and displays the predicted category.

## Dataset

For this project, data was scrapped from physics, biology, history, and computer science subject textbooks. The textbooks are in pdf files. The text of each textbook is extracted and stored as a text file in a folder with the subject names. The data is a list of the text files in different folders with each folder name as the name of the category. There are four categories in the dataset which will be the label; the categories are Physics, Computer Science, Biology, and History. 

References to the PDF files from which the text data was extracted:

### History
- [World History Textbook](https://www.livoniapublicschools.org/site/handlers/filedownload.ashx?moduleinstanceid=13448&dataid=5707&FileName=World%20History%20Textbook.pdf) 
- [The Book of History Volume 1 (1915)](http://www.public-library.uk/dailyebook/The%20Book%20of%20History%20Volume%201%20(1915).pdf)  
### Computer Science
- [Computer Science One](https://cse.unl.edu/~cbourke/ComputerScienceOne.pdf)
- [Computer Science - An Overview (12th Global Edition)](https://jhzhang.cn/resources/A050113G/Computer%20Science-%20An%20Overview%20(12th%20Global%20Edition).pdf)
- [Introduction to Computer Science Using Python](http://bedford-computing.co.uk/learning/wp-content/uploads/2015/10/Introduction-to-Computer-Science-Using-Python.pdf)
  
### Physics
- [University Physics with Modern Physics (13th Edition)](https://physica.cloud/ajab/uploads/2018/10/Hugh-D.-Young-Roger-A.-Freedman-A.-Lewis-Ford-University-Physics-with-Modern-Physics-with-MasteringPhysics®-13th-Edition-Addison-Wesley-2011.pdf)
- [Tipler and Llewellyn's Physics](https://web.pdx.edu/~pmoeck/books/Tipler_Llewellyn.pdf)
  
### Biology
- [Principles of Biology](http://dept.clcillinois.edu/biodv/PrinciplesOfBiology.pdf)
- [Raven Johnson McGraw-Hill Biology](https://biology.org.ua/files/lib/Raven_Johnson_McGraw-Hill_Biology.pdf)
  
Please note that the provided URLs may not be active or accessible in the future, as they are external resources.

## Usage

1. Clone or download the project repository to your local machine.
   - To download the code, visit the repository's page at https://github.com/IamDeeAnax/NLP_task. Click on the "Code" button and select "Download ZIP".
   - To clone the repository using git, open a terminal and navigate to the desired directory. Run the command "git clone https://github.com/IamDeeAnax/NLP_task" to create a local copy of the repository.

2. Open Jupyter Notebook and navigate to the project directory.

3. Open the .ipynb file containing the project code.
    - You can also click the "Open in colab" at the top of the code.
    
4. Follow the instructions provided in the notebook to run the project and explore its functionalities.

5. You can modify the code and experiment with different inputs to see the results.

### Running the Streamlit application

#### Installation

Before running the application, you will need to install the required Python packages. These packages are listed in the "requirements.txt" file in the "model" directory.

To install these packages, open a terminal (Mac or Linux) or command prompt (Windows) and navigate to the "model" directory using the "cd" command. For example: "cd path/to/model". Replace "path/to/model" with the actual path to the "model" directory on your computer.

Once you're in the "model" directory, run the following command to install the required packages: "pip install -r requirements.txt"

This will install all the necessary packages to run the model and the Streamlit application.

After installing the necessary packages, stay in the "model" directory and run: "streamlit run app.py"

The application should now be running at "localhost:8501" in your web browser.

## Contributing

Contributions to this project are welcome. If you encounter any issues, have suggestions, or want to contribute improvements, please follow these guidelines:

- Fork the repository and clone it to your local machine.
- Create a new branch for your contributions.
- Make your changes and test them thoroughly.
- Commit your changes and push them to your forked repository.
- Submit a pull request, describing the changes you have made.

Please ensure that your contributions adhere to the project's coding conventions and style guidelines.
