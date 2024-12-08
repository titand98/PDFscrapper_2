#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import PyPDF2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string

# Download stopwords if not already available
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Streamlit app
st.title("Sustainability Keyword Analysis")
st.subheader("Upload a PDF file to analyze text for sustainability-related keywords.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Preprocessing text
    text = re.sub(r'[^a-zA-Z0-9.,]', ' ', text)  # Remove special characters
    text = re.sub('[0-9]+', '', text)  # Remove numbers

    # Define keywords and stem them
    words = [
        'sustainability', 'sustainable', 'environment', 'climate', 'carbon',
        'renewable', 'green energy', 'recycling', 'biodiversity', 'emissions',
        'conservation', 'eco-friendly', 'solar', 'wind', 'energy', 'water', 'conservation'
    ]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in words]

    # Function to find sentences containing sustainability keywords
    def find_sustainability_sentences_with_stems(text, stems):
        sentences = re.split(r'(?<=[.!?]) +', text)
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = re.findall(r'\b\w+\b', sentence)
            stemmed_sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            if any(stem in stemmed_sentence_words for stem in stems):
                relevant_sentences.append(sentence)
        return relevant_sentences

    # Find sentences with sustainability keywords
    sustainability_sentences = find_sustainability_sentences_with_stems(text, stems)

    # Tokenize words
    all_sentences = " ".join(sustainability_sentences)
    sustain_words = word_tokenize(all_sentences)

    # Remove stopwords and punctuation
    en_stopwords = set(stopwords.words("english"))
    sustain_words_filter = [w for w in sustain_words if w.lower() not in en_stopwords and w not in string.punctuation]
    sustain_words_filter = [w for w in sustain_words_filter if len(w) > 2]

    # Calculate word frequency
    sustain_freq = FreqDist(sustain_words_filter)

    # Display frequent words as a table
    st.subheader("Frequent Words and Their Frequencies")
    freq_table = [{"Word": word, "Frequency": freq} for word, freq in sustain_freq.most_common(20)]
    st.table(freq_table)

    # Generate and display word cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(
        width=1000,
        height=500,
        stopwords=en_stopwords,
        colormap="plasma",
        collocations=False,
        max_words=700
    ).generate(" ".join(sustain_words_filter))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
else:
    st.info("Please upload a PDF file to begin.")


# In[ ]:




