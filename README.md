# company_description_generator


## Summary
I tried two main strategies:
  1. To use open source pre-trained models from huggingface.
  2. To use OpenAI API with the langchain library.
After various iterations, I decided to go with OpenAI GPT-4 due to the quality of its descriptions.

## Explanation of the preprocessing steps
For the open source strategy:
 1. Data was first cleaned using some basic cleaning operations using regex.
 2. It was then translated into the english language using the MarianMTModel open source model to have consistent output language.
 3. It was then compressed to roughly 1024 tokens using abstractive summarization, because this is the max input limit for Bart.
 4. Bart was then used to generate guided summaries.

For OpenAI:
 Seperate LangChains were employed to use GPT-4 to clean, translate and summarize the text with different prompts.

## Challenges
For the open source strategy:
  1. Since the models were running on my local machine, its speed was limited by my hardware. I used an NVidia GPU to speed it up.
  2. The quality of output summaries was a major hurdle. I tried different summarization models, but it seemed like they were largely ignoring the prompt and were focused on summarizing the inputs without focusing specifically on the problem, solution and target audience.

For OpenAI:
  1. The response time was extremely slow for GPT-3.5-turbo, so I used GPT-4.
  2. Since the response time was still not very high, I decided to parallelize the API calls. However, it was quickly rate limited, so I had to significantly reduce the frequency of API calls.

## Potential improvements
  1. Using abstractive summarization for text compression along with GPT-4 can significantly reduce run cost, and potentially still have similar quality to that without compression.
  2. This is a proof of concept version of the pipeline, for production, VertexAI on GCP would be more reliable and have better monitoring.


### NOTE:
1. The final version of the description generator is contained in: `src/langchain_summarizer_parallelized.py`
2. Previous versions are contained: `src/legacy_attempts`