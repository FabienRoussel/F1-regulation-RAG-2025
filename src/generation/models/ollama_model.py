"""Ollama model client for interacting with Ollama API."""

import requests
import json


class OllamaModel:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen3:4b"):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model_name: Name of the model to use (default: qwen3:4b)
        """
        self.base_url = base_url
        self.model_name = model_name

    def pull_model(self, model_name: str | None = None):
        """
        Pull a model from Ollama.

        Args:
            model_name: Name of the model to pull (default: uses instance model_name)
        """
        model = model_name or self.model_name
        url = f"{self.base_url}/api/pull"
        payload = {"name": model}

        print(f"Connecting to Ollama at {self.base_url}...")
        print(f"Pulling model: {model}")

        try:
            # Stream the response as pulling can take time
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")

                    # Show progress information
                    if "total" in data and "completed" in data:
                        total = data["total"]
                        completed = data["completed"]
                        percentage = (completed / total) * 100 if total > 0 else 0
                        print(f"\r{status}: {percentage:.1f}% ({completed}/{total} bytes)", end="", flush=True)
                    else:
                        print(f"\r{status}", end="", flush=True)

            print("\n✓ Model pulled successfully!")

        except requests.exceptions.ConnectionError:
            print(f"✗ Error: Could not connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running (try: docker-compose up ollama)")
        except requests.exceptions.RequestException as e:
            print(f"✗ Error pulling model: {e}")
    
    def generate(self, prompt: str, model_name: str | None = None, max_tokens: int = 2048) -> str:
        """
        Generate text using the Ollama model.

        Args:
            prompt: The input prompt to generate text from.
            model_name: Name of the model to use (default: uses instance model_name)
            max_tokens: Maximum number of tokens to generate (default: 2048)
        Returns:
            The generated text as a string.
        """
        model = model_name or self.model_name
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "options": {
                "temperature": 0
            },
            "stream": False  # Set to True if you want streaming responses
        }

        try:
            response = requests.post(url, json=payload)
            data = response.json()

            if "response" in data:
                return data["response"].strip()
            else:
                raise ValueError("No response found in the response.")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error generating text: {e}")
        except json.JSONDecodeError:
            raise ValueError("Error decoding JSON response from Ollama.")
        
    def is_model_available(self, model_name: str | None = None) -> bool:
        """
        Check if a model is available locally in Ollama.

        Args:
            model_name: Name of the model to check (default: uses instance model_name)
        Returns:
            True if the model is available, False otherwise.
        model = model_name or self.model_name
        """     

        model = model_name or self.model_name
        url = f"{self.base_url}/api/models"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            available_models = [m["name"] for m in data.get("models", [])]
            return model in available_models

        except requests.exceptions.ConnectionError:
            print(f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running.")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error checking models: {e}")
            return False
        except json.JSONDecodeError:
            print("Error decoding JSON response from Ollama.")
            return False
    
    
