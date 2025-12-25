import joblib
import streamlit as st

# Load saved files
final_df = joblib.load("final_df.joblib")
simi = joblib.load("similarity_matrix.joblib")

st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 English Book Recommendation System")


def recommend(book_title, max_results=2):
    # Case-insensitive match
    matches = final_df[
        final_df["title"].str.lower() == book_title.lower()
    ]

    if matches.empty:
        return None, "Book not found"

    idx = matches.index[0]

    # 🔐 Safety check to prevent IndexError
    if idx >= simi.shape[0]:
        return None, "Model data mismatch. Please retrain similarity matrix."

    similarity_scores = simi[idx]

    ranked_books = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in ranked_books:
        if i == idx or score <= 0:
            continue

        if i >= len(final_df):
            continue  # extra safety

        recommendations.append({
            "title": final_df.iloc[i]["title"],
            "score": round(float(score), 3)
        })

        if len(recommendations) == max_results:
            break

    if not recommendations:
        return None, "No similar books found"

    return recommendations, None


# ───────── UI ─────────
book_name = st.text_input("Enter any book name")

if st.button("Recommend"):
    if not book_name.strip():
        st.info("Please enter a book name 📘")
    else:
        results, error = recommend(book_name)

        if error:
            st.warning(error)
        else:
            st.subheader(f"Recommended Books (Found {len(results)})")
            for rec in results:
                st.write(f"📖 **{rec['title']}** — Similarity: `{rec['score']}`")
