import joblib
import streamlit as st

# -----------------------------
# Load saved files
# -----------------------------
final_df = joblib.load("final_df.joblib")
cv = joblib.load("count_vectorizer.joblib")
simi = joblib.load("similarity_matrix.joblib")

# Ensure index alignment and consistent text format
final_df = final_df.reset_index(drop=True)

# normalize titles
final_df["title"] = final_df["title"].astype(str).str.lower().str.strip()

st.title("📚 Book Recommendation System")

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(book_title):
    if not book_title:
        return None

    book_title = book_title.lower().strip()

    # match user input
    matches = final_df[final_df["title"] == book_title]

    # if book not found
    if matches.empty:
        return None

    # get positional index
    matched_pos = matches.index[0]

    # safety: check similarity matrix size
    if matched_pos >= len(simi):
        return None

    simil = simi[matched_pos]

    # get top 5 similar books
    book_list = sorted(
        list(enumerate(simil)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]  # skip itself

    recommendations = []
    for idx, score in book_list:
        if idx < len(final_df):
            recommendations.append(final_df.iloc[idx]["title"])

    return recommendations

# -----------------------------
# Streamlit UI
# -----------------------------
a = st.text_input("Enter any book name")

if st.button("Recommend"):
    results = recommend(a)

    if not a:
        st.info("Please enter a book name first 🙂")

    elif results is None or len(results) == 0:
        st.warning("Book not found or similarity index mismatch ❌")

    else:
        st.subheader("📖 Recommended Books:")
        for book in results:
            st.write("✔️", book)
