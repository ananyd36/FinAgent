from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import openai
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv("PHIDATA_API_KEY")

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
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals. Analyse and always give one stock of the day at the end of every query.",
    instructions=["Format your response using markdown and use tables to display data where possible. SHow charts if possible and share future predictions as well."],
    markdown=True
)

app = Playground(agents = [finance_agent, web_search_agent]).get_app()


if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)