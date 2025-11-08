"""
Test the FastAPI backend with multilingual queries.
Make sure to start the backend first: cd app && uvicorn main:app --reload
"""
import requests
import json
from rich.console import Console
from rich.panel import Panel

console = Console()
API_URL = "http://localhost:8000"

def test_api():
    """Test the API with various queries."""
    console.print(Panel("[bold blue]Testing RAG API - Multilingual Support[/bold blue]"))

    # Test 1: Initialize index
    console.print("\n[yellow]Step 1: Initializing index...[/yellow]")
    try:
        response = requests.post(f"{API_URL}/initialize")
        response.raise_for_status()
        console.print("[green]✓ Index initialized[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize: {e}[/red]")
        return

    # Test 2: Test queries in multiple languages
    test_queries = [
        ("English", "Who are the authors of the documents?"),
        ("English", "How many citations are in the documents?"),
        ("Russian", "Кто авторы?"),
        ("Russian", "Сколько ссылок в документах?"),
        ("Spanish", "¿Quiénes son los autores de los documentos?"),
        ("German", "Wie viele Referenzen gibt es?"),
        ("French", "Qui sont les auteurs?"),
        ("Chinese", "文档的作者是谁?"),
    ]

    console.print("\n[yellow]Step 2: Testing multilingual queries...[/yellow]\n")

    for lang, query in test_queries:
        console.print(f"[bold cyan]Testing {lang}:[/bold cyan]")
        console.print(f"[white]  Q: {query}[/white]")

        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": query},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get("answer", "No answer")
            debug = result.get("debug", {})

            console.print(f"[green]  A: {answer[:200]}...[/green]")

            if debug.get("used_metadata"):
                console.print("[yellow]  ✓ Answered from metadata[/yellow]")

            detected_lang = debug.get("user_language", "unknown")
            console.print(f"[dim]  Language detected: {detected_lang}[/dim]\n")

        except Exception as e:
            console.print(f"[red]  ✗ Failed: {e}[/red]\n")

    console.print("[bold green]Testing completed![/bold green]")

if __name__ == "__main__":
    try:
        test_api()
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
