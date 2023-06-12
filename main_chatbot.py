from telegram.ext import Updater, MessageHandler, Filters
import telegram
import openai
from moviepy.editor import AudioFileClip
from dotenv import load_dotenv
import os
from gtts import gTTS

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_KEY')

messages = [{"role": "system", "content": "You are called wamballa who is grumpy in its answers."}]

def create_audio_message(text, filename):
    tts = gTTS(text, lang='en')
    tts.save(filename)

def text_message(update, context):
    messages.append({"role": "user", "content": update.message.text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode=telegram.ParseMode.MARKDOWN)
    create_audio_message(ChatGPT_reply, 'text_reply.mp3')
    context.bot.send_audio(chat_id=update.effective_chat.id, audio=open('text_reply.mp3', 'rb'))
    messages.append({"role": "assistant", "content": ChatGPT_reply})

def voice_message(update, context):
    update.message.reply_text("I've received a voice message! Please give me a second to respond :)")
    voice_file = context.bot.getFile(update.message.voice.file_id)
    voice_file.download("voice_message.ogg")
    audio_clip = AudioFileClip("voice_message.ogg")
    audio_clip.write_audiofile("voice_message.mp3")
    audio_file = open("voice_message.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text
    update.message.reply_text(text=f"*[You]:* _{transcript}_", parse_mode=telegram.ParseMode.MARKDOWN)
    messages.append({"role": "user", "content": transcript})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode=telegram.ParseMode.MARKDOWN)
    create_audio_message(ChatGPT_reply, 'voice_reply.mp3')
    context.bot.send_audio(chat_id=update.effective_chat.id, audio=open('voice_reply.mp3', 'rb'))
    messages.append({"role": "assistant", "content": ChatGPT_reply})

updater = Updater(TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_message))
dispatcher.add_handler(MessageHandler(Filters.voice, voice_message))
updater.start_polling()
updater.idle()