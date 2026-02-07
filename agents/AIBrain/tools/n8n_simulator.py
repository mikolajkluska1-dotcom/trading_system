"""
n8n Simulator for Market News (v5.0)
====================================
Symuluje działanie automatyzacji n8n, która w produkcji przesyłałaby newsy via Webhook.
Tutaj pobieramy RSS i zapisujemy bufor dla Father Brian.
Wersja bez zewnętrznych zależności (używa standardowych bibliotek).
"""
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# Symulujemy bufor danych, który n8n zapisałby na dysku
N8N_BUFFER_FILE = "R:/Redline_Data/live_data/n8n_news_buffer.json"

def fetch_rss_titles(url):
    """Pobiera tytuły z RSS używając standardowych bibliotek"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            titles = []
            # Szukamy elementów 'item' i w nich 'title'
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                if title is not None:
                    titles.append(title.text)
            return titles
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return []

def fetch_real_news():
    print("🔄 n8n Simulator: Fetching RSS feeds (Native)...")
    sources = [
        "https://cointelegraph.com/rss",
        "https://cryptopanic.com/news/rss/"
    ]
    
    all_headlines = []
    for url in sources:
        titles = fetch_rss_titles(url)
        all_headlines.extend(titles)
            
    return ". ".join(all_headlines)

def update_buffer():
    news_text = fetch_real_news()
    if not news_text:
        news_text = "No news available (Connection error)."
        
    data = {
        "timestamp": str(datetime.now()),
        "news_summary": news_text,
        "source": "n8n_simulator_v1_native"
    }
    
    try:
        os.makedirs(os.path.dirname(N8N_BUFFER_FILE), exist_ok=True)
        with open(N8N_BUFFER_FILE, 'w') as f:
            json.dump(data, f)
        
        print(f"✅ n8n Buffer Updated: {news_text[:50]}...")
    except Exception as e:
        print(f"❌ Error updating buffer: {e}")
        
    return news_text

if __name__ == "__main__":
    update_buffer()
