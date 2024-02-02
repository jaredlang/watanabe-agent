import os 
import time
from dotenv import load_dotenv 
import json 
import requests 

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders.base import Document
from langchain_community.utilities import ApifyWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.reddit_search.tool import RedditSearchRun, RedditSearchSchema
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper

from langchain_openai import ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# import as an alias 
from pydantic import BaseModel as BaseModelv2, Field as Fieldv2 # pydantic v2 not compatible with langchain
from pydantic.v1 import BaseModel as BaseModelv1, Field as Fieldv1 # pydantic v1 compatible with langchain
#from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.tools import StructuredTool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser
)

from langchain.agents import AgentExecutor

import streamlit as st

from fastapi import FastAPI


# 0. Load env variables 
load_dotenv() 
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# 1. function for search
def search(query) -> str:
    """Search a query on Google for the content and the referenced URL"""
    print(f"Googling {query}...")
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print("SEARCH FOUND: \n", response.text)

    return response.text

# 2. function for scraping the website 
def scrape_website(objective:str, url: str) -> str: 

    print(f"Scraping website [{url}]...")
    apify = ApifyWrapper() 

    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": url}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )

    index = VectorstoreIndexCreator().from_loaders([loader])

    result = index.query_with_sources(objective)

    print("ANSWERS: \n", result["answer"])
    print("SOURCES: \n", result["sources"])

    output = result["answer"]
    if len(output) > 10000: 
        output = summarize(objective, output)

    return output

# 3. function for summarizing the website content
def summarize(objective, content):
    print("Summarizing...")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    print("SUMMARIZED: \n", output)

    return output

# 4. Tool for searching the internet 
searchTool = StructuredTool.from_function(
    func=search, 
    name="Search", 
    description="Search a query on Google for the content and the referenced URL. You should ask targeted questions"
)

# 4. Tool for scraping the website
class ScrapeWebsiteInput(BaseModelv1):
    """Inputs for Scraping a website on the given objective"""
    objective: str = Fieldv1(description="The objective & task that users give to the agent")
    url: str = Fieldv1(description="The url of the website to be scraped")

scrapeWebsiteTool = StructuredTool.from_function(
    func=scrape_website, 
    name="scrape_website",
    description="useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results",
    args_schema=ScrapeWebsiteInput, 
    return_direct=False
)

# redditSearchParam = RedditSearchSchema(
#     query="{input}", sort="new", time_filter="week", subreddit="all", limit="5"
# )

def search_reddit(objective: str) -> str: 
    redditWrapper = RedditSearchAPIWrapper(
        reddit_client_id=REDDIT_CLIENT_ID,
        reddit_client_secret=REDDIT_CLIENT_SECRET,
        reddit_user_agent="bot_agent",
    )

    posts = redditWrapper.run(
        query=objective, 
        sort="new", 
        time_filter="week", 
        subreddit="all", 
        limit=5
    )

    output = posts if len(posts) <= 4000 else summarize(objective, posts)

    return output


redditSearchTool = StructuredTool.from_function(
    func=search_reddit, 
    name="search-reddit", 
    description="""A tool that searches for posts on Reddit.
        Useful when you need to know post information on a subreddit.
        """
)

# Reddit Built-in Tool tends to return too much content (beyond token limits)
# RedditSearchRun(
#     api_wrapper=RedditSearchAPIWrapper(
#         reddit_client_id=REDDIT_CLIENT_ID,
#         reddit_client_secret=REDDIT_CLIENT_SECRET,
#         reddit_user_agent="bot_agent",
#     ),
#     # args_schema=redditSearchParam,
# )

tools = [searchTool, scrapeWebsiteTool, YahooFinanceNewsTool(), redditSearchTool]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm_with_tools = llm.bind_tools(tools)

MEMORY_KEY = "search_history"

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
        you do not make things up, you will try as hard as possible to gather facts & data to back up the research
        
        Please make sure you complete the objective above with the following rules:
        1/ You should do enough research to gather as much information as possible about the objective
        2/ If there are url of relevant links & articles, you will scrape it to gather more information
        3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
        4/ You should not make things up, you should only write facts & data that you have gathered
        5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        """
    ),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "search_history": lambda x: x["search_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def do_research(topic: str): 
    search_history = []

    result = agent_executor.invoke({
        "input": topic, 
        "search_history": search_history
    })

    print("RESULT: ", result)

    return result


def test(): 
    topic = "Why has Meta's Thread product grown more quickly than other products?"

    #search_result = search(topic)
    #searchTool.invoke(topic)
    #print(isinstance(ScrapeWebsiteInput, BaseModel))
    #print(type(ScrapeWebsiteInput))
    # scrape_website(
    #     objective=topic, 
    #     url="https://www.nytimes.com/2023/07/11/technology/threads-zuckerberg-meta-google-plus.html"
    # )
    # scrapeWebsiteTool.invoke({
    #     "objective": topic, 
    #     "url": "https://www.techtarget.com/whatis/feature/Meta-Threads-explained-Everything-you-need-to-know"
    # })
    result = do_research(topic)
    print(result)


def app(): 
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    st.header("AI research agent :bird:")
    topic = st.text_input("Research Topic:")

    placeholder = st.empty()

    if topic:
        start_time = time.time()
        placeholder.text(f"Doing research ...")

        result = do_research(topic)

        end_time = time.time()
        research_time = int(end_time - start_time)

        placeholder.text(f"Here is what I have found after [{research_time} seconds]:")

        st.info(result["output"])


if __name__ == "__main__": 
    # test()  # Local Testing 
    app()   # Streamlit webapp 

# FastAPI service 
# api = FastAPI()

# class Query(BaseModelv2): 
#     topic: str 

# @api.post("/")
# def service(query: Query): 
#     topic = query.topic
#     result = do_research(topic)
#     return result

