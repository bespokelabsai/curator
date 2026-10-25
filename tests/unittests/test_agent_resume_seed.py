from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bespokelabs.curator.agent.processor import MultiTurnAgenticProcessor
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.types.generic_response import GenericResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_turns", [0, 1, 2])
async def test_resumed_conversation_keeps_seed_and_next_speaker(tmp_path, completed_turns):
    requests = []
    interrupted = True

    async def call(request, session, status_tracker):
        if interrupted and len(requests) == completed_turns:
            raise RuntimeError("interrupted")
        requests.append(request.generic_request.messages)
        now = datetime.now(timezone.utc)
        return GenericResponse(
            response_message=f"reply {len(requests)}",
            raw_response={},
            generic_request=request.generic_request,
            created_at=now,
            finished_at=now,
        )

    def agent(name):
        return SimpleNamespace(
            name=name,
            model_name="gpt-4o-mini",
            is_completed=lambda response: False,
            prompt_formatter=PromptFormatter(
                model_name="gpt-4o-mini",
                prompt_func=lambda row: row["prompt"],
                system_prompt="Continue the conversation",
            ),
            _request_processor=SimpleNamespace(create_api_specific_request_online=lambda request: {}, call_single_request=call),
        )

    seeder, partner = agent("seeder"), agent("partner")
    with patch("bespokelabs.curator.agent.processor.AgentStatusTracker"):
        original = MultiTurnAgenticProcessor(seeder, partner, 3, "Opening task")
        with pytest.raises(RuntimeError, match="interrupted"):
            await original.run(str(tmp_path))
        interrupted = False
        resumed = MultiTurnAgenticProcessor(seeder, partner, 3, "Opening task")
        dataset = await resumed.run(str(tmp_path))

    assert len(dataset) == 3
    assert resumed.conversation_history == [
        {"role": "seeder", "content": "Opening task"},
        {"role": "partner", "content": "reply 1"},
        {"role": "seeder", "content": "reply 2"},
        {"role": "partner", "content": "reply 3"},
    ]
    system_message = {"role": "system", "content": "Continue the conversation"}
    expected_requests = [
        [system_message, {"role": "user", "content": "Opening task"}],
        [system_message, {"role": "assistant", "content": "Opening task"}, {"role": "user", "content": "reply 1"}],
        [
            system_message,
            {"role": "user", "content": "Opening task"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "reply 2"},
        ],
    ]
    assert requests[completed_turns] == expected_requests[completed_turns]
