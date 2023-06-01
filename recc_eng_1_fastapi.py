import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.cluster import KMeans
from fastapi import FastAPI, Request


demopref_data = pd.read_csv('/Actively Loaded Files/demopref_data_v2')
U_I_merged_df = pd.read_csv('/Actively Loaded Files/U_I_merged_df')
user_item_matrix = pd.read_pickle('/Actively Loaded Files/user_item_matrix.pkl')

# Loading the demopref kmeans model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)


# Create an instance of the Flask class
app1 = FastAPI()

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)





# Define a function to get the K most similar users
def find_similar_users(user_id, user_similarity, k=10):
    # Get the index of the user
    user_index = user_item_matrix.index.tolist().index(user_id)

    # Get the similarity scores for all users (convert to list to sort it)

    similarity_scores = user_similarity[user_index]

    # Sort the users based on their similarity scores

    sorted_indices = similarity_scores.argsort()[::-1]

    # Get the K most similar users
    similar_users = [list(U_I_merged_df['id_student'].unique())[i] for i in sorted_indices if i != user_index][:k]
    return similar_users




def student_exists(student_id, demopref_data):
    exists = demopref_data['id_student'].isin([student_id]).any()
    return exists 




def recommend_courses(student_id, demopref_data, user_similarity):
    if student_exists(student_id,demopref_data) == False:
        top_courses = demopref_data.iloc[:, :7].sum().dropna().sort_values(ascending=False).nlargest(10)
        print(f"Most Popular Courses : {top_courses}")
        return top_courses
    
    else:
        # select the cluster label of the target student
        target_student_cluster = demopref_data.loc[demopref_data['id_student'] == student_id, 'cluster_label'].values[0]

        # find the similar students based on VLE preferences
        similar_students = find_similar_users(student_id, user_similarity, k=15)

        # filter the similar students based on their cluster label

        similar_students = demopref_data.loc[(demopref_data['cluster_label'] == target_student_cluster 
                                              & demopref_data['id_student'].isin(similar_students))]

        # Find the courses that are most frequently taken by the filtered similar students
        freq_courses = similar_students.iloc[:, :7].sum().dropna().sort_values(ascending=False)
        freq_courses = freq_courses[freq_courses != 0].nlargest(3)

        # Recommend the courses that the target student has not taken yet 
        # but are among the most frequently taken courses by the filtered similar students
        target_student_courses = demopref_data.loc[demopref_data['id_student'] == student_id].iloc[0, :7]
        recommended_courses = freq_courses[~target_student_courses.astype(bool)].index.tolist()
        print(f"Recommended courses for student {student_id}: {recommended_courses}")

        return recommended_courses



# Define the API endpoint
@app1.get('/recommend-courses/{student_id}')
async def get_recommendations(student_id: int):

    # Call the recommend_courses function with the student ID, demopref_data, and user_similarity
    recommended_courses = recommend_courses(student_id, demopref_data, user_similarity)

    # Return the recommended courses as a JSON response
    return {'recommended_courses': recommended_courses}

