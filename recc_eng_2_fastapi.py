import pandas as pd
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi import FastAPI

app = FastAPI()


# Assuming that we're using AWS S3 Buckets which have versioning enabled 
# s3_client = boto3.client('s3')
# bucket_name = 'your-bucket-name'


# Function to get the latest object key with a specific pattern
#def get_latest_object_key(pattern):
#   response = s3_client.list_objects_v2(Bucket=bucket_name)
#   objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
#   latest_object_key = None
#   for obj in objects:
#       if re.match(pattern, obj['Key']):
#           latest_object_key = obj['Key']
#           break
#   if latest_object_key is None:
#       raise Exception(f"No object found in the S3 bucket with pattern: {pattern}")
#   return latest_object_key


### Download the latest similarity_df
##similarity_pattern = r'courses_similarity_df_v.*\.csv'
##latest_similarity_key = get_latest_object_key(similarity_pattern)
##with open("courses_similarity_df_latest.csv", "wb") as download_file:
##    s3_client.download_fileobj(bucket_name, latest_similarity_key, download_file)
##similarity_df = pd.read_csv("courses_similarity_df_latest.csv")
##


# Download the latest diverse_default_recommendations_list
##recommendations_pattern = r'diverse_default_recommendations_dict_spectral_v.*\.pkl'

##latest_recommendations_key = get_latest_object_key(recommendations_pattern)

##with open("diverse_default_recommendations_dict_spectral_latest.pkl", "wb") as download_file:
##    s3_client.download_fileobj(bucket_name, latest_recommendations_key, download_file)

##with open("diverse_default_recommendations_dict_spectral_latest.pkl", "rb") as file:
##    diverse_recommendations = pickle.load(file)



# Load the courses_similarity dataframe (this is built in in pandas, more efficient way to load pkl files :) )
 
file_name = '/Actively Loaded Files/courses_similarity_df.pkl'
similarity_df = pd.read_pickle(file_name)
distance_matrix = 1 - similarity_df

quantia_similarities_df = pd.read_pickle('/Actively Loaded Files/quantia_similarities_df.pkl')
quantia_dataset_cleaned = pd.read_pickle('/Actively Loaded Files/quantia_dataset_cleaned.pkl')

courses_data = pd.read_pickle('/Actively Loaded Files/preprocessed_df_withclusters.pkl')
with open("/Actively Loaded Files/diverse_default_recommendations_dict_spectral.pkl", "rb") as f:
    loaded_course_dict = pickle.load(f)



@app.get('/recommend-courses/IT/{course_id}')
async def get_recommendations(course_id: str):

    target_course_similarity = similarity_df.loc[course_id]
    top_5_similar_courses = target_course_similarity.sort_values(ascending=False).head(6)[1:]
    top_5_similar_courses = top_5_similar_courses.to_dict()

    return {'recommended_courses': top_5_similar_courses}

@app.get('/recommend-courses/MED/{course_id}')
async def get_recommendations(course_id: int):

    target_course_similarity = quantia_similarities_df.loc[course_id]
    top_3_similar_courses = target_course_similarity.sort_values(ascending=False).head(4)[1:]
    top_3_similar_courses_indices = top_3_similar_courses.index
    top_3_course_headlines = quantia_dataset_cleaned.loc[top_3_similar_courses_indices, 'HEADLINE']

    similarity_dict = {headline: similarity for headline, similarity in zip(top_3_course_headlines, top_3_similar_courses)}

    return {'recommended_courses': similarity_dict}

@app.get('/recommend-default-courses')
async def get_diverse_recommendations():
    return loaded_course_dict
