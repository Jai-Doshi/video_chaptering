from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader  
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_community.llms import GooglePalm, Ollama
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googletrans import Translator
import streamlit as st
import datetime

def get_available_languages(video_url):
    video_id = video_url.split('v=')[-1].split('&')[0]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = {transcript.language_code: transcript.language for transcript in transcript_list}
        return available_languages
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"Error for video ID {video_id}: {str(e)}")
        return {}

def video_details(video_url, language_code):
    video_id = video_url.split('v=')[-1].split('&')[0]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language_code]).fetch()
        if transcript:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True, language=language_code)
            docs = loader.load()
            global details 
            details = docs[0].metadata
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"Error for video ID {video_id}: {str(e)}")
        return []
    
    segments = []
    for entry in transcript:
        metadata = {
            'source': 'youtube',
            'video_id': video_id,
            'start_time': entry['start'],
            'duration': entry['duration']
        }
        translator = Translator()
        translation = translator.translate(entry['text'], dest='en')
        doc = Document(page_content=translation.text, metadata=metadata)
        segments.append(doc)
    
    return segments

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

# Streamlit UI
st.set_page_config(page_title='Video Chaptering', layout='wide')

st.title("YouTube Video Chaptering")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Generate Chapters"):
    with st.spinner('Generating Chapters for the YouTube Video . . .'):
        available_languages = get_available_languages(video_url)
        language_options = [f"{lang_code} ({lang_name})" for lang_code, lang_name in available_languages.items()]
        language_code = language_options[0].split()[0]
        segments = video_details(video_url, language_code)

        if segments:
            # Step 1: Split the Segments into Chunks
            combined_text = " ".join([doc.page_content for doc in segments])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(combined_text)

            # Step 2: Use Gemini LLM Model
            # llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key='AIzaSyCQgdRO8gDEpr-DUCqaCi0LOE_T4_Ns1OU', temperature=0.7)
            llm = Ollama(model='llama3')

            # Step 3: Create Prompt Template
            prompt_template = PromptTemplate(
                input_variables=["segment", "output_format"],
                template="Generate the chapters title for the following content:\n\n{segment}\n\n just give the titles no additional things required Title: and display in the output format for all the chapters as mentioned here {output_format}"
            )

            prompt_template = PromptTemplate(
                input_variables=["segment", "output_format"],
                template="""Generate the chapters title for the following transcript: {segment} just give the titles no additional things required and display in the output format for all the chapters as mentioned here
                Output Format:
                {output_format}

                Example Output:

                Chapter Title 1
                Chapter Title 2
                Chapter Title 3
                Please ensure that the output strictly adheres to the format provided, with no additional information beyond the chapter titles.
                """
            )

            # Step 4: Create the Chain
            chain = prompt_template | llm
            
            output_format = ("""
                    <li> chapter title </li>
                 """)
            response = chain.invoke({"segment": combined_text, "output_format": output_format})

            # Display the generated chapter titles
            st.markdown(f"<h1>{details['title']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2>Total Duration: {format_time(details['length'])}</h2>", unsafe_allow_html=True)
            st.image(f"{details['thumbnail_url']}")
            st.markdown(f"<h2>Chapters of {details['title']}</h2>", unsafe_allow_html=True)
            st.markdown(response, unsafe_allow_html=True)
        else:
            st.error("No transcript segments were found or processed.")
