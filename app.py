import os
import streamlit as st

# import the rag files
from rag_utility import process_document_to_chroma_db, answer_question

# setting working dir
working_dir = os.path.dirname(os.path.abspath((__file__)))


st.title("Simple Doc Answering App")

# file uploader widget
uploaded_file = st.file_uploader(
    "Upload a PDF file (best perf: less than 10pages)", type=["pdf"]
)


if uploaded_file is not None:
    # define save path
    save_path = os.path.join(working_dir, uploaded_file.name)

    # save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

        process_document = process_document_to_chroma_db(uploaded_file.name)
        st.info("Document Processed Succesfully")

        # to get users input
        user_question = st.text_area("Ask your question about the document")

        if st.button("Answer"):
            answer = answer_question(user_question)

            st.markdown("### LLM response")
            st.markdown(answer)
            st.info("letak source, page")
