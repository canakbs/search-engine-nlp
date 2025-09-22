from transformers import AutoTokenizer,AutoModel
import torch
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
eng_stopwords=stopwords.words("english") # defining stopwords(the, on, in...)

model_name="sentence-transformers/all-MiniLM-L6-v2" #Language Model
tokenizer=AutoTokenizer.from_pretrained(model_name) # to tokenize news titles
model=AutoModel.from_pretrained(model_name) # to embed the tokens

#Cleaning querry and news titles
def clean_text(text):
    text=re.sub(r"[^\w\s]","",text)
    text=re.sub(r"\d+"," ",text)
    text=text.lower()
    text_list=text.split()
    text_set=set([word for word in text_list if word not in eng_stopwords])
    text=" ".join([word for word in text_set if len(word)>2])
    return text



#Getting embedding of user input and news titles
def get_embedding(text):
    embeds=tokenizer(text,return_tensors="pt")
    with torch.no_grad():
        outputs=model(**embeds)
    last_hidden_state=outputs.last_hidden_state[:,0,:].numpy()
    embedding_mean=last_hidden_state.mean(axis=0)
    return embedding_mean 


# Example news data.(includes only the titles)
news_titles = [
    "Global Markets React to Federal Reserve Interest Rate Decision",
    "New AI Model Sets Record in Natural Language Processing Benchmarks",
    "Climate Change Summit in Geneva Brings World Leaders Together",
    "Breakthrough in Cancer Treatment Announced by Scientists",
    "Tech Giants Face Scrutiny Over Data Privacy Practices",
    "Historic Football Match Ends in Dramatic Penalty Shootout",
    "SpaceX Successfully Launches Next-Generation Satellite",
    "Breakthrough in Renewable Energy Storage Technology",
    "Study Finds Link Between Sleep and Mental Health",
    "Hollywood Prepares for Upcoming Awards Season",
    "Apple Unveils New iPhone with Advanced Camera Features",
    "Scientists Discover Water Traces on Mars’ Surface",
    "Global Protests Demand Action Against Climate Change",
    "New Cybersecurity Threat Targets Financial Institutions",
    "Elections in Europe See Record Voter Turnout",
    "Researchers Develop Faster COVID-19 Testing Method",
    "Major Airline Announces Expansion into Asia",
    "Tech Startups Compete in Annual Innovation Challenge",
    "UN Report Warns About Rising Sea Levels",
    "Breakthrough in Quantum Computing Achieved by IBM",
    "Tesla Introduces Affordable Electric Vehicle Model",
    "New Study Links Social Media Use to Anxiety in Teens",
    "Olympic Committee Confirms Paris 2024 Schedule",
    "Google Announces Major Updates to Search Algorithm",
    "Archaeologists Unearth Ancient City in Turkey",
    "World Bank Predicts Slower Global Economic Growth",
    "Researchers Find New Species in Amazon Rainforest",
    "Microsoft Expands Cloud Services Across Africa",
    "Famous Actor Announces Retirement from Hollywood",
    "Breakthrough in Alzheimer’s Research Gives New Hope"
]
titles_embed_list=[]
for text in news_titles:
    title=clean_text(text)
    title_embed=get_embedding(title)
    titles_embed_list.append(title_embed)



#Now we should combine all of them.
def main():
    query = input("What are you searching for? ")
    cleaned_query = clean_text(query)
    query_embed = get_embedding(cleaned_query)

    
    similarities = [
        (title, cosine_similarity(query_embed.reshape(1, -1), title_embedding.reshape(1, -1))[0][0])
        for title, title_embedding in zip(news_titles, titles_embed_list) #Matching titles and their embeddings
    ]

    
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

   
    top_5 = sorted_similarities[:5]

    print("\nTop 5 most similar news:")
    for title, score in top_5:
        print(f"{title} (similarity: {score:.4f})")

main()
#This is an example output
"""
What are you searching for? where is olympic comittee in 2024?

Top 5 most similar news:
Olympic Committee Confirms Paris 2024 Schedule (similarity: 0.8477)
Elections in Europe See Record Voter Turnout (similarity: 0.6305)
Global Protests Demand Action Against Climate Change (similarity: 0.6279)
Historic Football Match Ends in Dramatic Penalty Shootout (similarity: 0.6229)
New AI Model Sets Record in Natural Language Processing Benchmarks (similarity: 0.6069)

"""
    