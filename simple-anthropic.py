import anthropic

client = anthropic.Anthropic()

message_batch = client.beta.messages.batches.create(
    requests=[
        {
            "custom_id": "first-prompt-in-my-batch",
            "params": {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hey Claude, tell me a short fun fact about video games!",
                    }
                ],
            },
        },
        {
            "custom_id": "second-prompt-in-my-batch",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hey Claude, tell me a short fun fact about bees!",
                    }
                ],
            },
        },
    ]
)
print(message_batch)

"""
BetaMessageBatch(id='msgbatch_01XWYEcAqybHAWXqyinUyp8K', archived_at=None, cancel_initiated_at=None, created_at=datetime.datetime(2024, 12, 10, 21, 30, 23, 225753, tzinfo=datetime.timezone.utc), ended_at=None, expires_at=datetime.datetime(2024, 12, 11, 21, 30, 23, 225753, tzinfo=datetime.timezone.utc), processing_status='in_progress', request_counts=BetaMessageBatchRequestCounts(canceled=0, errored=0, expired=0, processing=2, succeeded=0), results_url=None, type='message_batch'
"""
