from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

class Transcript:

    def __init__(self, video_url):
        self.video_url = video_url

    def get_id(self):
        video_id = self.video_url.split('v=')[-1].split('&')[0]
        return video_id

    def get_video_information(self, language_code):
        try:
            loader = YoutubeLoader.from_youtube_url(
                self.video_url,
                add_video_info=True,
                language=language_code
            )
            docs = loader.load()
            return docs[0]
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            return "Transcript is not Available . . . ."

    def get_available_languages(self):
        video_id = self.get_id()
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_languages = {transcript.language_code: transcript.language for transcript in transcript_list}
            language_options = [f"{lang_code} ({lang_name})" for lang_code, lang_name in available_languages.items()]
            language_code = language_options[0].split()[0]
            return language_code
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            return str('en')

    def get_transcript(self):
        lang_code = self.get_available_languages()
        transcript = self.get_video_information(language_code=lang_code)
        return lang_code, transcript
