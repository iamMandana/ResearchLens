import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

def generate(
        prompt: str,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        timeout: int = 180,
        num_ctx: int = 4096,
) -> str:
    #send a prompt to Ollama and return the model's full reply as text.
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    #Stream=False asks Ollama to return one JSON object instead of many pieces.
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"num_ctx": num_ctx}}
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Ollama returned error ({error.code})."
            f"Is the model installed? Try: ollama pull {model}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not reach Ollama. Check that the Ollama app is running."
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Ollama did not answer within {timeout} seconds."
        ) from error

    return data["response"].strip()