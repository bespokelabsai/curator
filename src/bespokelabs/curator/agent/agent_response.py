from bespokelabs.curator.types.generic_response import GenericResponse


class AgentResponse(GenericResponse):
    """A response from an agent in a multi-turn conversation.

    This class extends the GenericResponse class to include agent-specific information,
    particularly the name of the agent that generated the response.

    Attributes:
        name (str): The name of the agent that generated this response.
    """

    name: str
