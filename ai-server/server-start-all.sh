#!/bin/bash
chmod +x server-start-all.sh
python get_news-web-app-only-intent.py &
python get_user_info-web-app-only-intent.py &
python web-app-ner-0POLMD.py 