#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pytesseract


# In[2]:


pip install opencv-python


# In[3]:


get_ipython().system('pip install pillow')


# # Extracting text from image

# In[4]:


import pytesseract
import matplotlib.pyplot as plt
from PIL import Image


# In[5]:


import cv2


# In[6]:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[7]:


img = cv2.imread('text1.png')


# In[8]:


plt.imshow(img)


# In[9]:


img2char = pytesseract.image_to_string(img)


# In[10]:


img2char


# In[ ]:





# # Document image analysis

# In[27]:


import cv2
import pytesseract
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

# Function to perform text summarization using NLTK
def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the sentences into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Calculate word frequency
    word_frequency = {}
    for word in stemmed_words:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in word_frequency.items():
            if word in sentence.lower():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq
                else:
                    sentence_scores[sentence] = freq

    # Get the summary by selecting top sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary = ' '.join(summary_sentences)

    return summary

# Example usage
image_path = "gemini image.png"
extracted_text = extract_text_from_image(image_path)

print("Extracted Text:")
print(extracted_text)

# Perform text summarization
summary = summarize_text(extracted_text)

print("\nSummary:")
print(summary)


# In[ ]:





# In[ ]:




