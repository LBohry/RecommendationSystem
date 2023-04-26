import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()


# Load the saved Word2Vec model from a pickle file
with open('wordembed_model.pkl', 'rb') as f1:
    wordembed_model = pickle.load(f1)
# Load the datafile from a pickle file
with open('preprocessed_df.pkl', 'rb') as f2:
    preprocessed_df = pickle.load(f2)



# Load the courses_similarity dataframe (this is built in in pandas, more efficient way to load pkl files :) )
 
file_name = 'courses_similarity_df.pkl'
similarity_df = pd.read_pickle(file_name)


@app.get('/recommend-courses/{course_id}')
async def get_recommendations(course_id: str):

    target_course_similarity = similarity_df.loc[course_id]
    top_20_similar_courses = target_course_similarity.sort_values(ascending=False).head(21)[1:]
    top_20_similar_courses = top_20_similar_courses.to_dict()

    return {'recommended_courses': top_20_similar_courses}


