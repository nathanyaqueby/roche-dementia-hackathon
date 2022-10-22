# import streamlit as st
from gtts import gTTS

text_en = "Gaby is your granddaughter, she is 33 now and she loves you very much. She also loves your cookies"
ta_tts = gTTS(text_en)
ta_tts.save("gaby_digimemoir.mp3")
audio_file = open("gaby_digimemoir.mp3", "rb")
audio_bytes = audio_file.read()
# st.audio(audio_bytes, format="audio/ogg", start_time=0)