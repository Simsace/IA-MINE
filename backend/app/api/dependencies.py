from fastapi import Header


async def optional_auth_dependency(authorization: str | None = Header(default=None)) -> dict:
    """Future-ready auth hook.

    Today the API is public for local development. Later this function can verify
    JWT tokens, API keys, Firebase Auth tokens, Supabase sessions, etc.
    """
    return {"authorization": authorization}

