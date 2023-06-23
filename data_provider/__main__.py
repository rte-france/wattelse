import typer

from data_provider.bing_news_provider import BingNewsProvider


if __name__ == "__main__":
    provider = BingNewsProvider()

    app = typer.Typer()

    @app.command("scrape")
    def scrape(keywords: str = typer.Argument(None, help="keywords for news search engine."),
               save_path: str = typer.Option(None, help="Path for writing results. File is in jsonl format.")
               ):
        results = provider.get_articles([keywords])
        provider.store_articles(results, save_path)

    @app.command("auto-scrape")
    def auto_scrape(
        inputfile_path: str = typer.Argument(
            None, help="path of jsonlines input file containing the urls."
        ),
        save_path: str = typer.Argument(None, help="Path for writting results."),
    ):
        print("Not implemented yet :-)")

    app()

