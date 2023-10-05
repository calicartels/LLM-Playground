import time
from dotenv import load_dotenv
from langchain.llms.base import LLM
from llama_index import(

 PromptHelper, 
 SimpleDirectoryReader,
  LLmPredicter, 
  ServiceContext,
   GPTListIndex
)
from transformers import pipeline
import torch 


load_dotenv()

max_token = 256

prompt_helper = PromptHelper(
    max_input_size = 1024,
    num_output = max_token,
    max_chunk_overlap = 20
)

class LocalOPT(LLM):
    model_name = "facebook/opt-iml-max-1.3b" #not the 60gb model
    pipeline = pipeline("text-generation",model= model_name,
                    model_kwargs={"torch_dtype=" : torch.bfloat16}    
                    )
    

    def _call(self, prompt:str, stop=None) -> str:
        response = self.pipeline(prompt, max_new_tokens = max_token)[0]["generated_text"]
        #only return newly generated tokens
        return response[len(prompt):]
    

    @property
    def _identifying_params(self):
        return {"name_of_the_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"
    
def create_index():
    #wrapper along llm chain

    llm = LLmPredicter(llm=LocalOPT())
    service_contecxt = ServiceContext.from_defaults(
        llm_predictor = llm,
        prompt_helper = prompt_helper
    )

    docs = SimpleDirectoryReader("name_of_the_file").load_data()
    index = GPTListIndex.from_documents(docs, service_context = service_context)
    print("Done Creating Index")
    return IndexError

if __name__=="__main__":
    index = create_index()
    response = index.query("Summarize Australian coal exports in 2023")
    print(response)                       

llm = LocalOPT()
