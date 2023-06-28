# Installation
If the installation of the pygooglenews dependency fails, please use the script `install_pygooglenews.sh`

# Description
Grabs news articles and store them as jsonlines file.

# Usage
```                                                                                                                                                 
(weak_signals) jerome@linux:~/dev/weak-signals$ PYTHONPATH=. python -m data_provider --help
                                                                                                                                                      
 Usage: python -m data_provider [OPTIONS] COMMAND [ARGS]...                                                                                           
                                                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]                           │
│ --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or customize the installation.    │
│                                                              [default: None]                                                                       │
│ --help                                                       Show this message and exit.                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ auto-scrape  Scrape data from Google or Bing news (multiple requests from a configuration file: each line of the file shall be compliant with the  │
│              following format: <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)                                    │
│ scrape       Scrape data from Google or Bing news (single request).                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 ```

# Remarques

You can expect a rate of 10-20% of articles not correctly processed
- problem of cookies management
- errors 404, 403
