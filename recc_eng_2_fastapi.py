import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI

app = FastAPI()


# Load the saved Word2Vec model from a pickle file
with open('wordembed_model.pkl', 'rb') as f1:
    wordembed_model = pickle.load(f1)
# Load the datafile from a pickle file
with open('preprocessed_df.pkl', 'rb') as f2:
    preprocessed_df = pickle.load(f2)



@app.get('/recommend-courses/{course_id}')
async def get_recommendations(course_id: str):

    # Get the embeddings for each course title and description
    target_course = preprocessed_df.loc[preprocessed_df['CourseId'] == course_id]['title_desc_cleaned'].tolist()
    target_embedding = np.concatenate([wordembed_model.wv[word] for word in target_course])

    max_len = max([len(np.concatenate([wordembed_model.wv[word] for word in row['title_desc_cleaned']])) for _, row in preprocessed_df.iterrows()])

    target_embedding_padded = pad_sequences([target_embedding.reshape(1,-1).T], maxlen=max_len, dtype='float32', padding='post')
    target_embedding_padded = target_embedding_padded.reshape(1,max_len)

    # Compute the similarity scores between the target course and all other courses in the dataset
    similarity_scores = {}
    for index, row in preprocessed_df.iterrows():
        course_id = row['CourseId']
        course_title_desc = row['title_desc_cleaned']
        course_embedding = np.concatenate([wordembed_model.wv[word] for word in course_title_desc])

        course_embedding_padded = pad_sequences([course_embedding], maxlen=max_len, dtype='float32', padding='post').reshape(1,max_len)
        similarity_scores[course_id] = float(cosine_similarity(target_embedding_padded, course_embedding_padded)[0][0])

    # Sort the courses by similarity score and return the top n courses
    similar_courses = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    return {'recommended_courses': similar_courses}


