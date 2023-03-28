from time import perf_counter
from langchain import OpenAI
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return index

def ask_bot(input_index = 'index.json'):
    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    while True:
        query = input('Enter the query: ')
        response = index.query(query, response_mode="compact")
        if response.response is not None:
            start_time = perf_counter()
            print("Bot:" + response.response)
            print(f"Time taken: {perf_counter() - start_time:.2f} seconds")
        else:
            print("\nSorry, I couldn't understand your question. Please try again.\n")
        

if __name__ == '__main__':
    # construct_index("./dataset/")
    ask_bot('index.json')
    
    