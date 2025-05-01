import typing as t
import aiofiles
import os

import aiohttp
import json
from tqdm import tqdm
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter

from bespokelabs.curator.request_processor.online.base_online_request_processor import GenericRequest
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.types.generic_response import GenericResponse

if t.TYPE_CHECKING:
    from bespokelabs.curator.agent.agent import Agent


class MultiTurnAgenticProcessor:
    def __init__(self, seeder: 'Agent', partner: 'Agent', total_steps: int, seed_message: str):
        self.seeder = seeder
        self.partner = partner
        self.total_steps = total_steps
        self.seed_message = seed_message

    async def run(self, working_dir: str):
        request_file = os.path.join(working_dir, 'responses_0.jsonl')
        async with aiohttp.ClientSession() as session:
            async with aiofiles.open(request_file, "a") as f:
                partener_request = self.partner.prompt_formatter.create_generic_request({'text': self.seed_message}, 0)
                for step in tqdm(range(self.total_steps), desc="Running MultiTurnAgenticProcessor"):
                    if step % 2 == 0:
                        partener_request = APIRequest(
                            task_id=step,
                            generic_request=partener_request,
                            api_specific_request=self.partner._request_processor.create_api_specific_request_online(partener_request),
                            attempts_left=1,
                            prompt_formatter=self.partner.prompt_formatter,
                        )
                        partener_response = await self.partner._request_processor.call_single_request(partener_request, session, status_tracker=None)
                        await self.append_response(self.partner.name, f, partener_response)
                        seeder_request = await self._unwrap_response_to_generic_resquest(partener_response, self.seeder, step)
                    else:
                        seeder_request = APIRequest(
                            task_id=step,
                            generic_request=seeder_request,
                            api_specific_request=self.seeder._request_processor.create_api_specific_request_online(seeder_request),
                            attempts_left=1,
                            prompt_formatter=self.seeder.prompt_formatter,
                        )
                        seeder_response = await self.seeder._request_processor.call_single_request(seeder_request, session, status_tracker=None)
                        await self.append_response(self.seeder.name, f, seeder_response)
                        partener_request = await self._unwrap_response_to_generic_resquest(seeder_response, self.partner, step)
        dataset_file = self.create_dataset_file(working_dir)

        return Dataset.from_file(dataset_file)

    async def append_response(self, name: str, f, response: GenericResponse):
        response = response.model_dump()
        response['name'] = name
        await f.write(json.dumps(response, default=str) + "\n")
            
    async def _unwrap_response_to_generic_resquest(self, response: GenericResponse, agent: 'Agent', step: int):
        return agent.prompt_formatter.create_generic_request({'text': response.response_message}, step)
    
    def create_dataset_file(self, working_dir: str):
        response_file = os.path.join(working_dir, 'responses_0.jsonl')
        dataset_file = os.path.join(working_dir, 'dataset.arrow')
        
        with ArrowWriter(path=dataset_file) as writer:
            with open(response_file, "r") as f_in:
                for line in f_in:
                    response = GenericResponse.model_validate_json(line)
                    row = response.model_dump()
                    # Write the row to the arrow file
                    writer.write(row)
            
            # Finalize the writer
            writer.finalize()
        
        return dataset_file


