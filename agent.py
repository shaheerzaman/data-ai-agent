import os
import time
import pathlib
import uuid
import requests
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import ollama
from tqdm import tqdm

load_dotenv()

model_name = os.getenv("OLLAMA_MODEL")


def init_ollama():
    """Initialize and return the model via Ollama, with a real progress bar."""
    try:
        base_url = os.getenv("OLLAMA_BASE_URL")
        ollama_client = ollama.Client(host=base_url)
        print(
            f"Pulling {model_name} model. This may take a while if not already present"
        )
        current_digest, bars = "", {}
        for progress in ollama_client.pull(model_name, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get("status"))
                continue

            if digest not in bars and (total := progress.get("total")):
                bars[digest] = tqdm(
                    total=total,
                    desc=f"pulling {digest[7:19]}",
                    unit="B",
                    unit_scale=True,
                )

            if completed := progress.get("completed"):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest

        print(f"{model_name} model is ready")
        return True

    except Exception as e:
        print(f"Error initializing Olama model {e}")
        return False


# Tool to read csv files
def read_csv(ctx: RunContext[str]):
    """
    Read a CSV file.

    Args:
        ctx (RunContext[str]): The context of the run.

    Returns:
        list[dict]: A list of dictionaries with the following keys:
            - product: str
            - sales: float
            - amount: float
    """
    print("tool called: read_csv")

    csv_binary = ctx.deps.data
    filepath = f"temp_{uuid.uuid4()}.csv"
    with open(filepath, "wb") as f:
        f.write(csv_binary)
    df = pd.read_csv(filepath)
    os.remove(filepath)

    return df.to_csv(orient="records")


def get_regional_data_from_database(products: list[str]) -> list[dict] | list:
    """
    Query the database for the products regional data.

    Args:
        products (list[str]): The products to query the database for.

    Returns:
        list[dict]: The results of the query.
    """
    print("Tool called: get_regional_data_from_database")

    try:
        connection_string = os.getenv("DATABASE_URL")
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                product_str = ",".join(f"'{product}'" for product in products)
                cur.execute(
                    f"SELECT region, product, sales, amount FROM regional_data WHERE product in {product_str}"
                )
                results = cur.fetchall()
                results_dict = []
                for row in results:
                    results_dict.append(
                        {
                            "region": row[0],
                            "product": row[1],
                            "sales": row[2],
                            "amount": row[3],
                        }
                    )

                return results_dict
    except Exception as e:
        print(f"Error getting regional data from database {e}")
        return []


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """
    Get the exchange rate.

    Args:
        from_currency (str): The currency to get the exchange rate from.
        to_currency (str): The currency to get the exchange rate to.

    Returns:
        float: The exchange rate from the from_currency to the to_currency.
    """
    print("tool called: get_exchange_rate")

    try:
        url = os.getenv("FREE_CURRENCY_API_URL") + "/v1/latest"
        api_key = os.getenv("FREE_CURRENCY_API_KEY")
        response = requests.get(
            url,
            params={"base_currency": from_currency, "currencies": to_currency},
            headers={"api_key": api_key},
        )
        return response.json()["data"][to_currency]
    except Exception as e:
        print(f"Error getting exchange rate: {e}")
        return None


def create_agent(system_prompt: str) -> Agent:
    """create the agent"""
    base_url = os.getenv("OLLAMA_BASE_URL") + "/v1"

    ollama_model = OpenAIModel(
        model_name=model_name, provider=OpenAIProvider(base_url=base_url)
    )

    # create the tools
    tools = [read_csv, get_regional_data_from_database, get_exchange_rate]

    # creaet the agent
    agent = Agent(ollama_model, tools=tools, system_prompt=system_prompt)

    return agent


async def run_agent(log_file: str, file_path: pathlib.Path):
    # System prompt
    system_prompt = """
            You are a helpful assistant that can read CSV files, query the database and get the exchange rate.
            You can use 'read_csv' tool to read the CSV file.
            You can use 'get_regional_data_from_database' tool to get the regional data from the database.
            You can use 'get_exchange_rate' tool to get the exchange rate.

            IMPORTANT: When reading a CSV file, you should:
            1. First read the CSV file using the read_csv tool
            2. Extract a list of product names from the CSV data
            3. Use those product names to query the database using get_regional_data_from_database tool
            4. Get the exchange rate from USD to EUR using get_exchange_rate tool
            5. Return the regional data in a table format with the exchange rate applied

            If the CSV doesn't contain product data, inform the user that they need to upload a CSV with product information.
    """

    # Create the messages
    messages = [
        """Read the CSV file with a tool and extract a list of products,
            get the regional data for those products from the database with a tool,
            finally get the exchange rate from USD to EUR with a tool,
            and finally return the regional data in a table format with the exchange rate applied.

            IMPORTANT: You MUST call all three tools in sequence:
            6. read_csv - to read the CSV file
            7. get_regional_data_from_database - to get regional data for the products
            8. get_exchange_rate - to get USD to EUR exchange rate

            If the CSV doesn't contain product names, inform the user they need to upload a CSV with product data."""
    ]

    deps = BinaryContent(data=file_path.read_bytes, media_type="text/csv")

    agent = create_agent(system_prompt)

    with open(log_file, "w") as log:
        log.write("-" * 100 + "\\n")
        log.write("Running agent \n\n")
        log.write("-" * 100 + "\\n")
        log.flush()

        start_time = time.time()
        async with agent.iter(messages, deps=deps) as agent_run:
            async for node in agent_run:
                log.write(str(node) + "\n\n")
                log.flush()

        end_time = time.time()
        log.write("-" * 100 + "\n\n")
        log.write(f"Agent run completed in {end_time - start_time} seconds\\n")
        log.write("-" * 100 + "\\n\\n")
        log.write(str(agent_run.result.output) + "\n\n")
        log.flush()
