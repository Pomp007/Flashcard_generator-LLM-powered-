import openai

# Set your API key
openai.api_key = "your-api-key-here"

def ask_question(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=150,
            temperature=0.7
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Error: {e}"

# Test example
if __name__ == "__main__":
    while True:
        user_input = input("Ask a question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = ask_question(user_input)
        print("Answer:", response)
