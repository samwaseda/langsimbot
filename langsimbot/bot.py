# This example show how to use inline keyboards and process button presses
import telebot
from llm import get_executor
from collections import defaultdict
import os


messages = defaultdict(list)


with open("BOT_API", "r") as f:
    TELEGRAM_TOKEN = f.read().split("\n")[0]


bot = telebot.TeleBot(TELEGRAM_TOKEN)


@bot.message_handler(func=lambda message: True)
def message_handler(message):
    messages[message.chat.id].append(("human", message.text))
    response = get_output(messages[message.chat.id])
    output = response['output']
    messages[message.chat.id].append(("ai", output))
    bot.send_message(message.chat.id, output)


bot.infinity_polling()


def get_output(messages):
    env = os.environ
    agent_executor = get_executor(
        api_provider=env.get("LANGSIM_PROVIDER", "OPENAI"),
        api_key=env.get("LANGSIM_API_KEY"),
        api_url=env.get("LANGSIM_API_URL", None),
        api_model=env.get("LANGSIM_MODEL", None),
        api_temperature=env.get("LANGSIM_TEMP", 0),
    )
    return list(agent_executor.stream({"conversation": messages}))[-1]
