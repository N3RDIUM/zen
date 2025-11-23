import os
from ollama import ChatResponse, Client, web_search, web_fetch
from tts import speak
from microphone import rec_from_mic
from logger import logger # TODO

SYSTEM = """You are Zen.
A minimalist AI voice assistant. English only.
Remember, you are a VOICE assistant. What you write is going to be SPOKEN to a user, and they probably won't read it.
Use tools if asked for even the simplest info, do not answer from your own knowledge, as you can hallucinate.

All numbers and must be converted to words for the text-to-speech program. 
All abbreviations must have spaces between each letter. For eg:
03:00 -> Three Hours (we use a 24-hour time format)
18:42 -> Eighteen Fourty Two Hours
ADC -> A D C
$42 -> Fourty-two dollars

And now for the most important rule.
Your responses will be short. Enough to be spoken within 5 to 10 seconds at a normal pace.
"""

def get_time() -> str:
    """
    Returns the current time.

    Args:
        None

    Returns:
        str: the current time in 24-hour format
    """

    return "03:00"

available_functions = {
    "get_time": get_time,
    "web_search": web_search,
    "web_fetch": web_fetch,
}
tools = list(available_functions.values())
model = "gpt-oss:20b"
messages = [{
    "role": "system",
    "content": SYSTEM
}]

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

while True:
    messages.append({
        "role": "user",
        "content": rec_from_mic()
    })

    while True:
        # Nested while loops are kinda scary.
        # But recursion is scarier.

        response: ChatResponse = client.chat(
            model,
            messages=messages,
            tools=tools,
            think=True
        )
        messages.append(response.message)
        print(messages[-1])

        calls = response.message.tool_calls
        if calls:
            for tool in calls:
                if function := available_functions.get(tool.function.name):
                    output = function(**tool.function.arguments)
                    messages.append({
                        "role": "tool",
                        "content": str(output),
                        "tool_name": tool.function.name
                    })
                    print(messages[-1])
                else:
                    messages.append({
                        "role": "tool",
                        "content": f"Tool not found: {tool.function.name}",
                        "tool_name": "not_found"
                    })
                    print(messages[-1])
        else:
            break

    print(messages[-1]["content"])
    speak(messages[-1]["content"])

