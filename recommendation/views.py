from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from django.templatetags.static import static
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
from django.http import JsonResponse

# Create your views here.
def index(request):
    data = pd.read_csv("static/dataset.csv", delimiter=";")

    content_df = data[['id', 'judul', 'deskripsi']]
    content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    user_id = int(request.GET.get('user_id'))
    book_id = int(request.GET.get('book_id'))
    top_n = int(request.GET.get('n', 5))

    recommendations = get_hybrid_recommendations(user_id, book_id, top_n, content_df, data)
    result = []
    for i, recommendation in enumerate(recommendations):
        result.append(int(recommendation))
    return JsonResponse({
         "status": 200,
         "result": result
    },safe=False)

def get_content_based_recommendations(book_id, top_n, content_df):
        # Use TF-IDF vectorizer to convert content into a matrix of TF-IDF features
        tfidf_vectorizer = TfidfVectorizer()
        content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])
        content_similarity = linear_kernel(content_matrix, content_matrix)
        index = content_df[content_df['id'] == book_id].index[0]
        similarity_scores = content_similarity[index]
        similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
        recommendations = content_df.loc[similar_indices, 'id'].values
        return recommendations

def get_collaborative_filtering_recommendations(user_id, top_n, data):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 
                                    'id', 
                                    'rating']], reader)
    
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    return recommendations

def get_hybrid_recommendations(user_id, book_id, top_n, content_df, data):
    content_based_recommendations = get_content_based_recommendations(book_id, top_n, content_df)
    # print("RESULT CONTENT BASED :", content_based_recommendations)
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n, data)
    # print("RESULT COLLABORATIVE :", collaborative_filtering_recommendations)
    hybrid_recommendations = list(set(content_based_recommendations)) + list(set(collaborative_filtering_recommendations))
    # print("RESULT HYBRID :", hybrid_recommendations)
    
    return hybrid_recommendations[:top_n]