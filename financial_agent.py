from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

#web search agent
web_search_agent = Agent(
    name = "Web Search Agent",
    role = 'Seach the wwb for the information',
    model = Groq(id='llama-3.3-70b-versatile'),
    tools = [DuckDuckGo()],
    instructions = ['Always include the sources'],
    show_tools_calls = True,
    markdown = True
)

#financial agent

finance_agent = Agent(
    name = 'Finance Agent',
    model = Groq(id = 'llama-3.3-70b-versatile'),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)


multi_ai_agent = Agent(
    team = [web_search_agent, finance_agent],
    instructions = ['Always include sources', "Use tables to display data"],
    show_tool_calls = True,
    markdown=True
)

multi_ai_agent.print_response("Summarise and analyse the stock for NVIDIA and share latest news in AI.")