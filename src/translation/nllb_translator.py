"""
NLLB-200 translation module for multilingual translation.
Meta's No Language Left Behind (NLLB) model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Optional, Dict
from loguru import logger
import time


# Language code mapping (Whisper -> NLLB)
WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
}


class NLLBTranslator:
    """Translator using NLLB-200 model."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda",
        target_lang: str = "tur_Latn"
    ):
        """
        Initialize NLLB translator.

        Args:
            model_name: HuggingFace model name
            device: Device to run on (cuda, cpu)
            target_lang: Target language code (NLLB format)
        """
        self.model_name = model_name
        self.device = device
        self.target_lang = target_lang

        logger.info(f"Loading NLLB model: {model_name}...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device)
            logger.info(f"Model moved to GPU")

        self.model.eval()

        load_time = time.time() - start_time
        logger.info(f"NLLB model loaded in {load_time:.2f}s")

    def translate(
        self,
        text: str,
        source_lang: str = "eng_Latn",
        target_lang: Optional[str] = None,
        max_length: int = 512
    ) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source_lang: Source language code (NLLB format)
            target_lang: Target language code (None to use default)
            max_length: Maximum length of translation

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""

        target = target_lang or self.target_lang

        # Skip translation if source and target are the same
        if source_lang == target:
            logger.debug(f"Same source and target language ({source_lang}), skipping translation")
            return text

        try:
            start_time = time.time()

            # Tokenize
            self.tokenizer.src_lang = source_lang
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode
            translation = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            translation_time = time.time() - start_time

            logger.info(
                f"Translated {len(text)} chars in {translation_time:.2f}s "
                f"({source_lang} -> {target})"
            )

            return translation

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text on error

    def translate_from_whisper_lang(
        self,
        text: str,
        whisper_lang: str,
        target_whisper_lang: str = "tr"
    ) -> str:
        """
        Translate using Whisper language codes.

        Args:
            text: Text to translate
            whisper_lang: Source language (Whisper format, e.g., "en")
            target_whisper_lang: Target language (Whisper format, e.g., "tr")

        Returns:
            Translated text
        """
        # Convert Whisper codes to NLLB codes
        source_lang = WHISPER_TO_NLLB.get(whisper_lang, "eng_Latn")
        target_lang = WHISPER_TO_NLLB.get(target_whisper_lang, "tur_Latn")

        return self.translate(text, source_lang, target_lang)

    def batch_translate(
        self,
        texts: list,
        source_lang: str = "eng_Latn",
        target_lang: Optional[str] = None
    ) -> list:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated texts
        """
        return [
            self.translate(text, source_lang, target_lang)
            for text in texts
        ]

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Get dictionary of supported languages."""
        return WHISPER_TO_NLLB.copy()


def test_translator():
    """Test translator functionality."""
    logger.info("Testing NLLB translator...")

    translator = NLLBTranslator(
        model_name="facebook/nllb-200-distilled-600M",
        device="cuda",
        target_lang="tur_Latn"
    )

    # Test English to Turkish
    test_texts = [
        "Hello, how are you?",
        "This is a test of the translation system.",
        "The meeting will start at 3 PM.",
        "Can you please send me the documentation?"
    ]

    logger.info("\n=== English to Turkish ===")
    for text in test_texts:
        translation = translator.translate(text, "eng_Latn", "tur_Latn")
        logger.info(f"EN: {text}")
        logger.info(f"TR: {translation}\n")

    # Test Turkish to English
    turkish_texts = [
        "Merhaba, nasılsın?",
        "Toplantı saat 15:00'te başlayacak.",
        "Lütfen bana belgeleri gönderir misin?"
    ]

    logger.info("\n=== Turkish to English ===")
    for text in turkish_texts:
        translation = translator.translate(text, "tur_Latn", "eng_Latn")
        logger.info(f"TR: {text}")
        logger.info(f"EN: {translation}\n")

    # Test with Whisper language codes
    logger.info("\n=== Using Whisper language codes ===")
    text = "Good morning, let's start the meeting."
    translation = translator.translate_from_whisper_lang(text, "en", "tr")
    logger.info(f"EN: {text}")
    logger.info(f"TR: {translation}")


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/translation_test.log")
    test_translator()
