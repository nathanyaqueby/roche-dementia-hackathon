# import streamlit as st
from gtts import gTTS

text_en = "This is a microwave, you can use it to heat your food. Would you like me to give you a step by step " \
          "instruction on how to use it?"
ta_tts = gTTS(text_en)
ta_tts.save("micro_digimemoir.mp3")
audio_file = open("micro_digimemoir.mp3", "rb")
audio_bytes = audio_file.read()
# st.audio(audio_bytes, format="audio/ogg", start_time=0)