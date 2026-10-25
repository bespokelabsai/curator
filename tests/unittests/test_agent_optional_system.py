from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bespokelabs.curator.agent.processor import MultiTurnAgenticProcessor
from bespokelabs.curator.llm.prompt_formatter import PromptFormatter


@pytest.mark.parametrize(
    "system_messages",
    [
        [],
        [{"role": "system", "content": "Be concise"}],
        [
            {"role": "system", "content": "Be concise"},
            {"role": "system", "content": "Use plain text"},
        ],
    ],
)
@pytest.mark.asyncio
async def test_agent_prompt_preserves_optional_system_messages(tmp_path, system_messages):
    seen = []

    async def call(request, session, status_tracker):
        seen.extend(request.generic_request.messages)
        raise RuntimeError("stop before network")

    def agent(name):
        return SimpleNamespace(
            name=name,
            model_name="gpt-4o-mini",
            is_completed=lambda response: False,
            prompt_formatter=PromptFormatter(
                model_name="gpt-4o-mini",
                prompt_func=lambda row: system_messages + [{"role": "user", "content": row["prompt"]}],
            ),
            _request_processor=SimpleNamespace(create_api_specific_request_online=lambda request: {}, call_single_request=call),
        )

    with patch("bespokelabs.curator.agent.processor.AgentStatusTracker"):
        processor = MultiTurnAgenticProcessor(agent("seeder"), agent("partner"), 1, "Opening task")
        with pytest.raises(RuntimeError, match="stop before network"):
            await processor.run(str(tmp_path))
    assert seen == system_messages + [{"role": "user", "content": "Opening task"}]
