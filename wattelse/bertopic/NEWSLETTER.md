# How to generate a newsletter using BERTopic

## Creation of configuration files

# `newsletter.cfg`

Follow the model in `bertopic/config/newsletters`

# `feed.cfg`

Follow the model in `bertopic/config/feeds`

### One-shot creation

```
(weak_signals) jerome@linux:~/dev/weak-signals$ CUDA_VISIBLE_DEVICES=0 python -m wattelse.bertopic newsletter --help

                                                                                                                                                      
 Usage: python -m wattelse.bertopic newsletter [OPTIONS] NEWSLETTER_CFG_PATH                                                                          
                                               DATA_FEED_CFG_PATH                                                                                     
                                                                                                                                                      
 Creates a newsletter associated to a data feed.                                                                                                      
                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    newsletter_cfg_path      PATH  Path to newsletter config file [default: None] [required]                                                      │
│ *    data_feed_cfg_path       PATH  Path to data feed config file [default: None] [required]                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```


### Periodic creations

TBD

