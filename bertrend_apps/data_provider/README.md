# Installation
If the installation of the pygooglenews dependency fails, please use the script `install_pygooglenews.sh`

# Description
Grabs articles (news or scientific articles) and store them as jsonlines file.

Several data providers are supported:
- Arxiv
- Google News
- Bing News
- NewsCatcher

# Usage
```                                                                                                                                                 
(weak_signals) jerome@linux:~/dev/weak-signals$ python -m bertrend_apps.data_provider --help
                                                                                                                                                      
                                                                                                                                                       
 Usage: python -m data_provider [OPTIONS] COMMAND [ARGS]...                                                                                           
                                                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]                           │
│ --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or customize the installation.    │
│                                                              [default: None]                                                                       │
│ --help                                                       Show this message and exit.                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ auto-scrape          Scrape data from Google or Bing news (multiple requests from a configuration file: each line of the file shall be compliant   │
│                      with the following format: <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)                   │
│ generate-query-file  Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results. This will │
│                      split the broad query into many ones, each one covering an 'interval' (range) in days covered by each atomic request. If you  │
│                      want to cover several keywords, run the command several times with the same output file.                                      │
│ scrape               Scrape data from Google or Bing news (single request).                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 ```

```
(weak_signals) jerome@linux:~/dev/weak-signals$ python -m bertrend_apps.data_provider --help auto-scrape --help
                                                                                                                                                      
Usage: python -m data_provider auto-scrape [OPTIONS] [REQUESTS_FILE]                                                                                 
                                                                                                                                                      
 Scrape data from Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the    
 following format: <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)                                                   
 Parameters ---------- requests_file: str     Text file containing the list of requests to be processed provider: str     News data provider. Current 
 authorized values [google, bing, newscatcher] save_path: str     Path to the output file (jsonl format)                                              
 Returns -------                                                                                                                                      
                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   requests_file      [REQUESTS_FILE]  path of jsonlines input file containing the expected queries. [default: None]                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --max-results        INTEGER  maximum number of results per request [default: 50]                                                                  │
│ --provider           TEXT     source for news [google, bing, newscatcher] [default: google]                                                        │
│ --save-path          TEXT     Path for writing results. [default: None]                                                                            │
│ --help                        Show this message and exit.                                                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


```

```
(weak_signals) jerome@linux:~/dev/weak-signals$ python -m bertrend_apps.data_provider generate-query-file --help
                                                                                                                                                      
 Usage: python -m data_provider generate-query-file [OPTIONS] [KEYWORDS]                                                                              
                                                                                                                                                      
 Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results. This will split the broad query  
 into many ones, each one covering an 'interval' (range) in days covered by each atomic request. If you want to cover several keywords, run the       
 command several times with the same output file.                                                                                                     
 Parameters ---------- keywords: str     query described as keywords after: str     "from" date, formatted as YYYY-MM-DD before: str     "to" date,   
 formatted as YYYY-MM-DD save_path: str     Path to the output file (jsonl format)                                                                    
 Returns -------                                                                                                                                      
                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   keywords      [KEYWORDS]  keywords for news search engine. [default: None]                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --after            TEXT     date after which to consider news [format YYYY-MM-DD] [default: None]                                                  │
│ --before           TEXT     date before which to consider news [format YYYY-MM-DD] [default: None]                                                 │
│ --save-path        TEXT     Path for writing results. File is in jsonl format. [default: None]                                                     │
│ --interval         INTEGER  Range of days of atomic requests [default: 30]                                                                         │
│ --help                      Show this message and exit.                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

# Remarques

You can expect a rate of 10-20% of articles not correctly processed
- problem of cookies management
- errors 404, 403
