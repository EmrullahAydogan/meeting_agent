"""
DeepSeek AI client for meeting analysis and content understanding.
"""

import os
from openai import OpenAI
from typing import List, Dict, Optional
from loguru import logger
import time


class DeepSeekAnalyzer:
    """AI analyzer using DeepSeek API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize DeepSeek analyzer.

        Args:
            api_key: DeepSeek API key (from env if None)
            base_url: API base URL
            model: Model name (deepseek-chat, deepseek-reasoner)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment or parameters")

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        logger.info(f"DeepSeek analyzer initialized: {model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Response text
        """
        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )

            response_time = time.time() - start_time
            content = response.choices[0].message.content

            logger.info(f"DeepSeek response received in {response_time:.2f}s")

            return content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return ""

    def extract_topics(self, transcript: str, language: str = "auto") -> List[str]:
        """
        Extract main topics from meeting transcript.

        Args:
            transcript: Meeting transcript text
            language: Language of transcript (auto, en, tr)

        Returns:
            List of topics
        """
        if not transcript.strip():
            return []

        prompt = f"""Analyze the following meeting transcript and extract the main topics discussed.
Return ONLY a numbered list of topics, one per line.

Language: {language}
Transcript:
{transcript}

Topics:"""

        messages = [
            {"role": "system", "content": "You are an expert meeting analyst. Extract key topics concisely."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.3)

        # Parse topics
        topics = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering and formatting
                topic = line.lstrip('0123456789.-* ')
                if topic:
                    topics.append(topic)

        logger.info(f"Extracted {len(topics)} topics")
        return topics

    def summarize(self, transcript: str, language: str = "auto", max_sentences: int = 5) -> str:
        """
        Generate summary of meeting transcript.

        Args:
            transcript: Meeting transcript text
            language: Language of transcript
            max_sentences: Maximum sentences in summary

        Returns:
            Summary text
        """
        if not transcript.strip():
            return ""

        lang_instruction = ""
        if language == "tr":
            lang_instruction = "Respond in Turkish."
        elif language == "en":
            lang_instruction = "Respond in English."

        prompt = f"""Summarize the following meeting transcript in {max_sentences} sentences or less.
{lang_instruction}

Transcript:
{transcript}

Summary:"""

        messages = [
            {"role": "system", "content": "You are an expert meeting summarizer. Create concise, informative summaries."},
            {"role": "user", "content": prompt}
        ]

        summary = self.chat(messages, temperature=0.5, max_tokens=500)

        logger.info(f"Generated summary ({len(summary)} chars)")
        return summary

    def extract_action_items(self, transcript: str, language: str = "auto") -> List[str]:
        """
        Extract action items and tasks from transcript.

        Args:
            transcript: Meeting transcript text
            language: Language of transcript

        Returns:
            List of action items
        """
        if not transcript.strip():
            return []

        lang_instruction = ""
        if language == "tr":
            lang_instruction = "Respond in Turkish."
        elif language == "en":
            lang_instruction = "Respond in English."

        prompt = f"""Extract all action items, tasks, and to-dos from the following meeting transcript.
Return ONLY a numbered list, one action item per line.
{lang_instruction}

Transcript:
{transcript}

Action Items:"""

        messages = [
            {"role": "system", "content": "You are an expert at identifying action items in meetings."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.3)

        # Parse action items
        actions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                action = line.lstrip('0123456789.-* ')
                if action:
                    actions.append(action)

        logger.info(f"Extracted {len(actions)} action items")
        return actions

    def generate_research_queries(self, topics: List[str], language: str = "en") -> List[str]:
        """
        Generate research queries based on topics.

        Args:
            topics: List of topics to research
            language: Language for queries

        Returns:
            List of search queries
        """
        if not topics:
            return []

        topics_text = "\n".join(f"- {topic}" for topic in topics)

        prompt = f"""Generate 2-3 concise search queries to research the following topics.
Return ONLY the search queries, one per line, in {language}.

Topics:
{topics_text}

Search Queries:"""

        messages = [
            {"role": "system", "content": "You are an expert at generating effective search queries."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.5, max_tokens=300)

        # Parse queries
        queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.endswith(':'):
                query = line.lstrip('0123456789.-* ')
                if query:
                    queries.append(query)

        logger.info(f"Generated {len(queries)} research queries")
        return queries

    def analyze_sentiment(self, transcript: str) -> str:
        """
        Analyze sentiment of meeting.

        Args:
            transcript: Meeting transcript text

        Returns:
            Sentiment description
        """
        if not transcript.strip():
            return "neutral"

        prompt = f"""Analyze the sentiment/tone of the following meeting transcript.
Respond with ONE word: positive, negative, or neutral.

Transcript:
{transcript[:1000]}

Sentiment:"""

        messages = [
            {"role": "system", "content": "You are a sentiment analysis expert."},
            {"role": "user", "content": prompt}
        ]

        sentiment = self.chat(messages, temperature=0.1, max_tokens=10).strip().lower()

        logger.info(f"Sentiment: {sentiment}")
        return sentiment


def test_deepseek():
    """Test DeepSeek analyzer."""
    logger.info("Testing DeepSeek analyzer...")

    analyzer = DeepSeekAnalyzer()

    # Test transcript
    transcript = """
    Good morning everyone. Today we need to discuss the Q4 product roadmap.
    First, we should finalize the mobile app features. John, can you prepare
    the technical specifications by Friday? Also, we need to review the budget
    for the marketing campaign. Sarah will send the updated numbers by EOD.
    Let's schedule a follow-up meeting next Tuesday to review progress.
    """

    # Test topic extraction
    logger.info("\n=== Topics ===")
    topics = analyzer.extract_topics(transcript, "en")
    for i, topic in enumerate(topics, 1):
        logger.info(f"{i}. {topic}")

    # Test summarization
    logger.info("\n=== Summary ===")
    summary = analyzer.summarize(transcript, "en", max_sentences=3)
    logger.info(summary)

    # Test action items
    logger.info("\n=== Action Items ===")
    actions = analyzer.extract_action_items(transcript, "en")
    for i, action in enumerate(actions, 1):
        logger.info(f"{i}. {action}")

    # Test research queries
    logger.info("\n=== Research Queries ===")
    queries = analyzer.generate_research_queries(topics, "en")
    for i, query in enumerate(queries, 1):
        logger.info(f"{i}. {query}")

    # Test sentiment
    logger.info("\n=== Sentiment ===")
    sentiment = analyzer.analyze_sentiment(transcript)
    logger.info(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/deepseek_test.log")
    test_deepseek()
