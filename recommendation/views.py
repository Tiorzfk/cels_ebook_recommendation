from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from django.templatetags.static import static
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
from django.http import JsonResponse
from .forms import UploadFileForm
from django.views.decorators.csrf import csrf_exempt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

# Create your views here.
def index(request):
    data = pd.read_csv("static/export_dataset.csv", delimiter=";")

    content_df = data[['id', 'judul', 'deskripsi']]
    content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    user_id = int(request.GET.get('user_id'))
    book_id = int(request.GET.get('book_id'))
    top_n = int(request.GET.get('n', 5))

    # try:
    recommendations = get_hybrid_recommendations(user_id, book_id, top_n, content_df, data)
    result = []
    for i, recommendation in enumerate(recommendations):
        result.append(int(recommendation))
    return JsonResponse({
        "status": 200,
        "result": result
    },safe=False)
    # except:
    #     return JsonResponse({
    #         "status": 500,
    #         "result": []
    #     },safe=False)
    
def stemming(content):
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # stemming process
    sentence = content
    output   = stemmer.stem(sentence) 

    print("STEMMING ", output)  

    return output

def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")

def get_content_based_recommendations(book_id, top_n, content_df):
        # Preprocessing Data pada kolom 'deskripsi'
        # Mengisi nilai kosong (NaN) dalam kolom 'deskripsi' dengan string kosong ''
        content_df['Content'] = content_df['Content'].fillna('')

        # Mengubah teks deskripsi menjadi lowercase (huruf kecil) untuk konsistensi dalam perhitungan TF-IDF
        content_df['Content'] = content_df['Content'].apply(lambda x: _normalize_whitespace(x.lower()))

        # Menghapus karakter khusus dan angka menggunakan regular expression pada kolom 'deskripsi'
        # Punctuation Removing
        content_df['Content'] = content_df['Content'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        # print("BEFORE STEMMING = ", content_df['Content'])
        # # Stemming
        # content_df['Content'] = content_df['Content'].apply(lambda x: stemming(x))
        # print("AFTER STEMMING = ", content_df['Content'])

        # STOPWORD
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        content_df['Content'] = content_df['Content'].apply(lambda x: stopword.remove(x))

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

@csrf_exempt
def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
            return JsonResponse({
                "status": 200,
                "result": []
            },safe=False)
    else:
        form = UploadFileForm()

    return JsonResponse({
        "status": 400,
        "result": []
    },safe=False)

def handle_uploaded_file(f):
    with open("static/dataset.csv", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)