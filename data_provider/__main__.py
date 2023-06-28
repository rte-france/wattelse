import typer

from data_provider.bing_news_provider import BingNewsProvider
from data_provider.google_news_provider import GoogleNewsProvider
from loguru import logger

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("scrape")
    def scrape(
        keywords: str = typer.Argument(None, help="keywords for news search engine."),
        provider: str = typer.Option("google", help="source for news [bing, google]"),
        after: str = typer.Option(
            None, help="date after which to consider news [format YYYY-MM-DD]"
        ),
        before: str = typer.Option(
            None, help="date before which to consider news [format YYYY-MM-DD]"
        ),
        save_path: str = typer.Option(
            None, help="Path for writing results. File is in jsonl format."
        ),
    ):
        """Scrape data from Google or Bing news (single request).

        Parameters
        ----------
        keywords: str
            query described as keywords
        provider: str
            News data provider. Current authorized values [google, bing]
        after: str
            "from" date, formatted as YYYY-MM-DD
        before: str
            "to" date, formatted as YYYY-MM-DD
        save_path: str
            Path to the output file (jsonl format)

        Returns
        -------

        """
        if provider == "bing":
            provider = BingNewsProvider()
        else:
            provider = GoogleNewsProvider()
        results = provider.get_articles(keywords, after, before)
        provider.store_articles(results, save_path)

    @app.command("auto-scrape")
    def auto_scrape(
        requests_file: str = typer.Argument(
            None, help="path of jsonlines input file containing the expected queries."
        ),
        provider: str = typer.Option("google", help="source for news [bing, google]"),
        save_path: str = typer.Option(None, help="Path for writing results."),
    ):
        """Scrape data from Google or Bing news (multiple requests from a configuration file: each line of the file shall be compliant with the following format:
        <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)

        Parameters
        ----------
        requests_file: str
            Text file containing the list of requests to be processed
        provider: str
            News data provider. Current authorized values [google, bing]
        save_path: str
            Path to the output file (jsonl format)

        Returns
        -------

        """
        if provider == "bing":
            provider = BingNewsProvider()
        else:
            provider = GoogleNewsProvider()
        logger.info(f"Opening query file: {requests_file}")
        with open(requests_file) as file:
            try:
                requests = [line.rstrip().split(";") for line in file]
            except:
                logger.error("Bad file format")
                return -1
            results = provider.get_articles_batch(requests)
            logger.info(f"Storing {len(results)} articles")
            provider.store_articles(results, save_path)

    app()
