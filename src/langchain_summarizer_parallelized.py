import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm_name = "gpt-4"
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.3, model=llm_name, openai_api_key=openai_api_key)

INPUT_FILE_PATH = './data/task_sources.csv'
OUTPUT_FILE_PATH = './data/company_descriptions.csv'


def process_text(index, text, chain):
    """Helper function to process text using specified chain."""
    return index, chain.run(text)


def execute_chain_in_parallel(df_column, chain, max_rate=10):
    """Execute a specified chain function in parallel on a DataFrame column."""
    # Predefine the list to store the results
    results_list = [None] * len(df_column)

    # Create a ThreadPoolExecutor to run the chain function in parallel
    with ThreadPoolExecutor(max_workers=max_rate) as executor:
        # Start time to ensure rate limit
        start_time = time.time()
        # Execute the chain function on each text in the DataFrame column
        futures = [executor.submit(process_text, index, text, chain) for
                   index, text in enumerate(df_column)]
        # Collect the results as they become available
        for future in futures:
            index, result = future.result()
            results_list[index] = result
        # End time to ensure rate limit
        end_time = time.time()
        # Calculate the time taken to process the requests
        elapsed_time = end_time - start_time
        # If the requests were processed in less than a minute, sleep for the remaining time
        if elapsed_time < 60:
            time.sleep(60 - elapsed_time)

    return results_list


def preprocess(input_file_path: str) -> pd.DataFrame:
    """
    Reads data from a CSV file, cleans and translates the text in the DataFrame.

    Parameters:
    input_file_path (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: A DataFrame with the cleaned and translated text.
    """
    # Import data
    df = pd.read_csv(input_file_path)

    # Set up cleaning chain
    cleaning_prompt = ChatPromptTemplate.from_template(
        "Clean the following text ```{text}```"
    )
    cleaning_chain = LLMChain(llm=llm, prompt=cleaning_prompt)

    # Set up translating chain
    translating_prompt = ChatPromptTemplate.from_template(
        "Translate the following text into english ```{text}```"
    )
    translating_chain = LLMChain(llm=llm, prompt=translating_prompt)

    # Process text
    preprocess_chain = SimpleSequentialChain(
        chains=[cleaning_chain, translating_chain],
        verbose=True
    )
    cleaned_descriptions = execute_chain_in_parallel(df["text"],
                                                     preprocess_chain)

    # Store results
    df["cleaned_text"] = cleaned_descriptions
    df.to_csv("./data/cleaned_descriptions.csv", index=False)
    return df

def generate_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summaries from the cleaned and translated text in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the cleaned and translated text.

    Returns:
    pd.DataFrame: A DataFrame containing the generated summaries.
    """
    # Set up summarizing chain
    summarizing_prompt = ChatPromptTemplate.from_template(
        "Extract summary in the following format enclosed by single quotes"
        "'PROBLEM: describe the problem the company is trying to solve "
        "SOLUTION: company's proposed solution "
        "TARGET USERS: target users of the company "
        "OTHER DETAILS: other important details of the company', "
        "for the following company description enclosed by triple backticks "
        "```{company_description}```"
    )
    summarizing_chain = LLMChain(llm=llm, prompt=summarizing_prompt)

    # Generate summaries
    summaries = execute_chain_in_parallel(df["cleaned_text"], summarizing_chain)

    # Store results
    df["summaries"] = summaries
    return df[["summaries"]]


if __name__ == "__main__":
    print("preprocessing")
    cleaned_df = preprocess(INPUT_FILE_PATH)
    print("finished preprocessing")
    print("summarizing")
    summaries_df = generate_summaries(cleaned_df)
    print("finished summarizing")
    summaries_df.to_csv(OUTPUT_FILE_PATH, index=False)