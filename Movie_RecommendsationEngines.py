from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
df=pd.read_csv("D:/files/movie_dataset.csv")
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

cv= CountVectorizer()

text=["saketh prabhir prabhir ","saketh saketh prabhir"]
count_matrix= cv.fit_transform(text)
print(cv.get_feature_names())

print(count_matrix.toarray())


from sklearn.metrics.pairwise import cosine_similarity

similarity_scores= cosine_similarity(count_matrix)

print(similarity_scores)

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 50 movies
i=0
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>50:
			break

