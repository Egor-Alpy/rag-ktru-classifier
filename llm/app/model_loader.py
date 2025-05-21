from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from loguru import logger


class ModelLoader:
    """Загрузчик модели LLM"""

    def __init__(
            self,
            model_id: str,
            model_revision: str = "main",
            quantization: str = "4bit"
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.quantization = quantization
        self.model = None
        self.tokenizer = None

    async def load_model(self):
        """Загружает модель и токенизатор"""
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        # Загрузка модели выполняется в отдельном потоке, чтобы не блокировать сервер
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor, self._load_model_sync
            )

    def _load_model_sync(self):
        """Синхронная загрузка модели"""
        try:
            logger.info(f"Загрузка токенизатора для модели {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                use_fast=True,
                trust_remote_code=True
            )

            logger.info(f"Загрузка модели {self.model_id}...")
            # Настройка параметров квантизации
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.quantization == "8bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                quantization_config = None

            # Загрузка модели с учетом квантизации
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "revision": self.model_revision,
                "trust_remote_code": True,
                "device_map": "auto"
            }

            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            logger.info(f"Модель {self.model_id} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель"""
        return self.model is not None and self.tokenizer is not None

    async def generate(
            self,
            prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.1,
            top_p: float = 0.9,
            top_k: int = 50
    ) -> str:
        """Генерирует текст на основе промпта"""
        if not self.is_model_loaded():
            raise ValueError("Модель не загружена")

        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        # Генерация текста выполняется в отдельном потоке, чтобы не блокировать сервер
        with ThreadPoolExecutor() as executor:
            text = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self._generate_sync(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
            )

            return text

    def _generate_sync(
            self,
            prompt: str,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int
    ) -> str:
        """Синхронная генерация текста"""
        try:
            # Подготовка промпта
            # Предполагается специфический формат для моделей Saiga
            if "saiga" in self.model_id.lower():
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            else:
                formatted_prompt = prompt

            # Токенизация
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

            # Генерация
            with torch.no_grad():
                tokens = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Декодирование
            output = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

            # Извлечение только сгенерированного текста (ответа ассистента)
            if "saiga" in self.model_id.lower():
                # Для Saiga необходимо извлечь только часть ответа ассистента
                assistant_prefix = "<|assistant|>"
                if assistant_prefix in output:
                    output = output.split(assistant_prefix)[1].strip()
            else:
                # Для других моделей просто убираем входной промпт
                if formatted_prompt in output:
                    output = output.replace(formatted_prompt, "").strip()

            return output
        except Exception as e:
            logger.error(f"Ошибка при генерации текста: {e}")
            raise