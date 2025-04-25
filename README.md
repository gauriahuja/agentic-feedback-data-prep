# Agentic-feedback-data-prep
 Persuasive Dialogue Summarization for Feedback Agent Training

This project processes persuasive charity-related dialogue from the [Convokit "Persuasion for Good" dataset](https://convokit.cornell.edu/documentation/persuasionforgood.html) using Natural Language Processing (NLP). It generates structured conversation summaries to serve as input prompts for downstream agentic AI systems — especially feedback-driven agents trained to respond, coach, or recommend strategies based on conversation content.

## Objective
To extract, summarize, and structure persuasive conversations involving donation appeals, enabling their use in AI feedback training pipelines. This is a precursor step for building negotiation-aware AI agents that can evaluate and deliver targeted feedback based on the style, success, and context of persuasion.


## Technologies Used

- **Python 3.9+**
- **Hugging Face Transformers** (`facebook/bart-large-cnn` for summarization)
- **pandas** (data handling)
- **NLTK / regex** (optional for cleaning)


## Project Structure

```
persuasive-dialogue-nlp/
├── data/
│   ├── persuasionforgood_dataset.csv            # Raw dataset of persuasive dialogues
│   ├── charity_conversation_summary.txt         # Processed summaries for AI agent use
│
├── scripts/
│   ├── summarize_dialogues.py                   # Script to load, clean, and summarize dialogue
│
├── outputs/
│   ├── summarized_dialogue_output.txt           # Final output with summaries and full conversation
│
├── README.md


Installation
 Install dependencies:
   ```bash
   pip install pandas transformers

 How It Works

1. Load the dataset containing human dialogue (e.g., donation conversations).
2. Sort and clean the conversations using pandas.
3. Join the dialogue into a script format.
4. Use Hugging Face's BART summarization model to generate a concise summary.
5. Save the structured output as a `.txt` file containing:
   - Summary
   - Full cleaned conversation

Example Output

Summary of the Conversation:
The conversation includes polite greetings, followed by appeals to support children's charities. The donor is asked how much they would like to contribute and responds with a small donation. The interaction focuses on empathy and awareness.

Full Script:
Hello how are you
Can I tell you all
Very well, how about you?


Research Relevance
This project is part of ongoing work in AI feedback systems, where LLM-powered agents guide users through negotiation or persuasion tasks by analyzing dialogue. The summarized outputs here are ready to be fed into agentic AI systems for personalized coaching and ethical decision support.

License
This project is for academic and research purposes under the University of Florida AI Systems Research Initiative.

Author
Gauri Ahuja
M.S. in Computer Science, University of Florida  
[LinkedIn](https://linkedin.com/in/gauri777) | [Email](mailto:ahujagauri@ufl.edu)
