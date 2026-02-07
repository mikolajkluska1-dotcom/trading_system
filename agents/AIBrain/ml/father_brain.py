"""
Father Brain - Neuro-Symbolic AI Supervisor (v5.0)
==================================================
Logika oparta na LLM (OpenAI) do analizy sentymentu rynkowego i newsów.
Odbiera dane z symulatora n8n (lub webhooków w produkcji).
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class FatherBrain:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
    def ponder(self, agents_report, n8n_news_context):
        """
        Myśli nad strategią.
        agents_report: Tekstowe podsumowanie od dzieci (Scanner, Technician...)
        n8n_news_context: Newsy dostarczone przez n8n (lub symulator)
        
        Returns:
            tuple: (sentiment_score, strategic_advice)
        """
        if not self.client:
            return 0.0, "Father is offline (No API Key). Be careful."

        prompt = f'''
        ROLE: You are the 'Father', a veteran crypto macro-strategist.
        
        INPUT DATA:
        1. MARKET NEWS (via n8n): "{n8n_news_context}"
        2. FAMILY REPORTS (Technicals):
        {agents_report}
        
        TASK:
        - Analyze conflict between Technicals (Family) and Fundamentals (News).
        - If News are extremely bad (SEC lawsuit, war), OVERRIDE technical buy signals.
        - If News are neutral, trust your children.
        
        OUTPUT (JSON ONLY):
        {{
            "sentiment_score": float (-1.0 to 1.0),
            "strategic_advice": "One short sentence for Mother Brain."
        }}
        '''
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get('sentiment_score', 0.0), data.get('strategic_advice', 'No advice')
        except Exception as e:
            print(f"👴 Father Error: {e}")
            return 0.0, "Error connecting to wisdom."
