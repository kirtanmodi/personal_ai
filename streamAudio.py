from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello world! This is a streaming test. My name is OpenAI. I'm the best system out there",
)

response.stream_to_file("output.mp3")
