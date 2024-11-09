from bespokelabs import curator

poet = curator.Prompter(
    prompt_func=lambda: "Write a poem about the beauty of computer science",
    model_name="gpt-4o-mini",
)

poem = poet()
print(poem["response"][0])
