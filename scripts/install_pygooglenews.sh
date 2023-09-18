curl -O https://files.pythonhosted.org/packages/3f/d5/695ef6cd1da80e090534562ba354bc72876438ae91d3693d6bd2afc947df/pygooglenews-0.1.2.tar.gz
tar -xvzf pygooglenews-0.1.2.tar.gz
cd pygooglenews-0.1.2
pip install pygooglenews --no-deps
pip install feedparser --force
pip install beautifulsoup4 --force
pip install dateparser --force
pip install requests --force