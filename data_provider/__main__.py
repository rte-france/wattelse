import typer

from data_provider.bing_news_provider import BingNewsProvider
from data_provider.google_news_provider import GoogleNewsProvider

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("scrape")
    def scrape(keywords: str = typer.Argument(None, help="keywords for news search engine."),
               provider: str = typer.Option("bing", help="source for news [bing, google]"),
               save_path: str = typer.Option(None, help="Path for writing results. File is in jsonl format.")
               ):
        if provider == "bing":
            provider = BingNewsProvider()
        else:
            provider = GoogleNewsProvider()
        results = provider.get_articles([keywords])
        provider.store_articles(results, save_path)

    @app.command("auto-scrape")
    def auto_scrape(
        requests: str = typer.Argument(
            None, help="path of jsonlines input file containing the expected queries."
        ),
        save_path: str = typer.Argument(None, help="Path for writing results."),
    ):
        print("Not implemented yet :-)")

    app()

