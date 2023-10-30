import openai

api_key = "use your api you butpalz"
while True:
  x = input("You: ")  # Get user input
  if x=="exit()":
    break
  openai.api_key = api_key

    # Create a chat conversation with user input
  conversation = [
      {"role": "user", "content": x}
  ]

    # Generate a response using GPT-3.5 Turbo
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation
  )

  # Extract and print the model's reply
  reply = response.choices[0].message["content"]
  print("AI: " + reply)


  
