import joblib
import streamlit as st

# Load saved files
final_df = joblib.load("final_df.joblib")
cv = joblib.load("count_vectorizer.joblib")
simi = joblib.load("similarity_matrix.joblib")

st.title("📚 Book Recommendation System")

def recommend(book_title):
    matched_book = final_df[final_df["title"] == book_title.lower()].index
    if len(matched_book) == 0:
        return None
    
    matched_book = matched_book[0]
    simil = simi[matched_book]

    book_list = sorted(
        list(enumerate(simil)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []
    for i in book_list:
        recommendations.append(final_df.iloc[i[0]]["title"])
    
    return recommendations


# User input
a = st.text_input("Enter any book name")

# Button
if st.button("Recommend"):
    if a:
        results = recommend(a)
        if results is None:
            st.warning("Book not found ❌")
        else:
            st.subheader("Recommended Books:")
            for book in results:
                st.write("📖", book)
    else:
        st.info("Please enter a book name")
