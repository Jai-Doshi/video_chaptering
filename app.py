import streamlit as st

from transcript import format_time, Transcript
from translation import Translation
from chaptering import Chaptering


def main():
    st.set_page_config(page_title='Video Chaptering', page_icon='assets/logo.png', layout='wide')

    col1, col2 = st.columns([1, 9])
    with col1:
        st.image('assets/logo.png')
    with col2:
        st.title("Video Chaptering By VIDCRAZE")

    video_url = st.text_input("Enter YouTube Video URL")

    if st.button('Generate Chapters'):
        with st.spinner('Generating Chapters for the YouTube Video . . .'):
            st.write('Processing Started')
            processing_url = Transcript(video_url)
            lang_code, transcript = processing_url.get_transcript()
            st.write('Transcript Generated')
            translation = Translation(
                transcript=transcript.page_content,
                source=lang_code
            )
            translated_text = translation.get_translated_text()
            st.write(f'Translated into {lang_code}')
            chaptering = Chaptering(text=translated_text)
            chapters = chaptering.get_chapters()
            st.write('Created Chapters Successfully')
            details = transcript.metadata
            st.markdown(f"<h1>{details['title']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2>Total Duration: {format_time(details['length'])}</h2>", unsafe_allow_html=True)
            st.image(f"{details['thumbnail_url']}")
            st.markdown(f"<h2>Chapters of {details['title']}</h2>", unsafe_allow_html=True)
            st.markdown(chapters, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
