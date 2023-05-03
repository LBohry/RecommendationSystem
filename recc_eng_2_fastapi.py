import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()


# Assuming that we're using AWS S3 Buckets which have versioning enabled 

##response = s3_client.list_objects_v2(Bucket=bucket_name)
##objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
##
##latest_object_key = None
##for obj in objects:
##    if re.match(r'courses_similarity_df_v.*\.csv', obj['Key']):
##        latest_object_key = obj['Key']
##        break
##if latest_object_key is None:
##    raise Exception("No similarity dataframe found in the S3 bucket")
##
##
##
##with open("courses_similarity_df_latest.csv", "wb") as download_file:
##    s3_client.download_fileobj(bucket_name, latest_object_key, download_file)
##
##
##
##similarity_df = pd.read_csv("courses_similarity_df_latest.csv")



# Load the courses_similarity dataframe (this is built in in pandas, more efficient way to load pkl files :) )
 
file_name = 'courses_similarity_df.pkl'
similarity_df = pd.read_pickle(file_name)


@app.get('/recommend-courses/{course_id}')
async def get_recommendations(course_id: str):

    target_course_similarity = similarity_df.loc[course_id]
    top_20_similar_courses = target_course_similarity.sort_values(ascending=False).head(21)[1:]
    top_20_similar_courses = top_20_similar_courses.to_dict()

    return {'recommended_courses': top_20_similar_courses}


