import pandas as pd
from transformers import pipeline #imports the pipeline function from the hugging face transformers library, to load pretrained AI-models like bart, gpt etc. in just 1 line of code.

#  Step 1: Load the dataset
file_path = "300_dialog.csv"  # dataset file name
df = pd.read_csv(file_path)
# it loads the .csv file as table so we can work with it in python and we need to access the dialogue data to prepare a script.

# Step 2: Create conversation script 
df = df.sort_values(by="Turn")  # Sort by turn , it sorts based on the turn so that the conversation flows in the correct order.
dialogue_lines = df["Unit"].dropna().tolist() # extracts the unit column which has the dialogue lines , removes any empty ones , and converts it into a list, to get a clean list of conversation lines we can work with .
conversation_script = "\n".join(dialogue_lines) # combines all the dialogue lines into one big conversation with each line on a new row. to turn it into a readable script and feed it to a summarizer.

# Step 3: Summarize conversation 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") # loads a pre-trained summarization model from hugging face , it automatically generates a short summary of the full dialogue.

if len(conversation_script) > 2000: # if the script is longer than 2000 characters it cut's it short so the model can handle it becasue most of the summarizer(especially BART) have the token/input limit.
    conversation_script = conversation_script[:2000]

summary = summarizer(conversation_script, max_length=100, min_length=30, do_sample=False)[0]["summary_text"] # sends the script to the model and get's back the summary string, this how AI tells us what the conversation is in few sentences.

# Step 4: Final Output  # formats the summary + full dialogue into a nicely structured
final_prompt = f"""  
Summary of the Conversation:
{summary} # so we can save it as a proper output file that looks good

Full Script:
{conversation_script}
"""

#  Step 5: Save to a file 
with open("generated_script_output.txt", "w", encoding="utf-8") as f:
    f.write(final_prompt) #createsa .txt file and writes the final formatted script into it.

print("Script extracted and saved to generated_script_output.txt")
#prints the message at the end to let yoou know it worked

