import joblib
import streamlit as st

final_df = joblib.load("final_df.joblib")
simi = joblib.load("similarity_matrix.joblib")

st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 English Book Recommendation System")


def recommend(book_title, max_results=10):
    matches = final_df[
        final_df["title"].str.lower() == book_title.lower()
    ]

    if matches.empty:
        return None, 0, "Book not found"

    idx = matches.index[0]

    if idx >= simi.shape[0]:
        return None, 0, "Model data mismatch"

    similarity_scores = simi[idx]

    ranked_books = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    total_matches = 0
    recommendations = []

    for i, score in ranked_books:
        if i == idx or score <= 0:
            continue

        total_matches += 1

        if len(recommendations) < max_results:
            recommendations.append({
                "title": final_df.iloc[i]["title"],
                "score": round(float(score), 3)
            })

    if total_matches == 0:
        return None, 0, "No similar books found"

    return recommendations, total_matches, None


book_name = st.text_input("Enter any book name")

if st.button("Recommend"):
    if not book_name.strip():
        st.info("Please enter a book name 📘")
    else:
        results, count, error = recommend(book_name)

        if error:
            st.warning(error)
        else:
            st.subheader(f"Total similar books found: {count}")
            st.subheader(f"Showing top {len(results)} recommendations")

            for rec in results:
                st.write(f"📖 **{rec['title']}** — Similarity: `{rec['score']}`")
