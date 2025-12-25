import joblib
import streamlit as st
import numpy as np

final_df = joblib.load("final_df.joblib")
simi = joblib.load("similarity_matrix.joblib")

st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 English Book Recommendation System")


def recommend(book_title, max_results=2):
    
    matched_book = final_df[
        final_df["title"].str.lower() == book_title.lower()
    ].index

    if len(matched_book) == 0:
        return None

    idx = matched_book[0]
    similarity_scores = simi[idx]

    # Sort books by similarity score (descending)
    ranked_books = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in ranked_books:
        if i == idx:
            continue  # skip the same book
        if score <= 0:
            continue  # ignore zero similarity

        recommendations.append({
            "title": final_df.iloc[i]["title"],
            "score": round(float(score), 3)
        })

        if len(recommendations) == max_results:
            break

    if len(recommendations) == 0:
        return None

    return recommendations


book_name = st.text_input("Enter any book name")

if st.button("Recommend"):
    if not book_name.strip():
        st.info("Please enter a book name 📘")
    else:
        results = recommend(book_name)

        if results is None:
            st.warning("Book not found or similarity too low ❌")
        else:
            st.subheader(f"Recommended Books (Found {len(results)})")

            for rec in results:
                st.write(f"📖 **{rec['title']}**  — Similarity: `{rec['score']}`")
