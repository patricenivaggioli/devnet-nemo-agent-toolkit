import asyncio
import sys

from nat.runtime.loader import load_workflow
from nat.utils.type_utils import StrPath


async def get_callable_for_workflow(config_file: StrPath):
    """
    Creates an end-to-end async callable which can run a NAT workflow.

    Note that this "yields" the callable so you have to access via an
    asynchronous generator:

      async for callable in get_callable_for_workflow(..)):
          # use callable here

    Args:
        config_file (StrPath): a valid path to a NAT configuration file

    Yields:
        The callable
    """
    # load a given workflow from a configuration file
    async with load_workflow(config_file) as workflow:

        # create an async callable that runs the workflow
        async def single_call(input_str: str) -> str:

            # run the input through the workflow
            async with workflow.run(input_str) as runner:
                # wait for the result and cast it to a string
                return await runner.result(to_type=str)

        yield single_call


async def amain():
    async for callable in get_callable_for_workflow(sys.argv[1]):
        # read queries from stdin and process them serially
        query_num = 1
        try:
            while True:
                query = input()
                result = await callable(query)
                print(f"Query {query_num}: {query}")
                print(f"Result {query_num}: {result}")
                query_num += 1
        except EOFError:
            pass


asyncio.run(amain())
