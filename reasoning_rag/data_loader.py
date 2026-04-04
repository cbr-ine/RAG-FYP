"""数据加载模块"""
import logging
from datasets import load_dataset
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, random_seed: int = 42):
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.random_seed = random_seed  # 固定种子，保证每次分割一致

    def load_bioasq_dataset(self, train_ratio: float = 0.8):
        """加载BioASQ数据集并手动分割训练/测试集"""
        logger.info("Loading BioASQ dataset from HuggingFace...")

        try:
            ds = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
            available_splits = list(ds.keys())
            logger.info(f"Available splits: {available_splits}")

            main_split = available_splits[0]
            full_dataset = ds[main_split]
            logger.info(f"Loaded {len(full_dataset)} examples from '{main_split}' split")

            indices = list(range(len(full_dataset)))
            random.seed(self.random_seed)   # 固定随机种子
            random.shuffle(indices)

            split_point = int(len(indices) * train_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            self.train_data = [full_dataset[i] for i in train_indices]
            self.test_data = [full_dataset[i] for i in test_indices]

            logger.info(
                f"Dataset split: Train size: {len(self.train_data)}, "
                f"Test size: {len(self.test_data)}"
            )

            self.dataset = ds

            logger.info("\nSample data structure:")
            if self.train_data:
                sample = self.train_data[0]
                logger.info(f"Keys: {list(sample.keys())}")
                logger.info(f"Question: {sample.get('question', 'N/A')[:100]}...")
                logger.info(f"Answer: {sample.get('answer', 'N/A')[:100]}...")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Creating empty dataset placeholders...")
            self.train_data = []
            self.test_data = []

    def get_passages(self, split: str = 'train', max_passages: int = None) -> List[Dict]:
        """
        从数据集中提取段落用于构建索引。

        每条 Q&A 数据生成两种 passage：
          1. 纯答案 passage          —— text = answer
          2. 问答组合 passage        —— text = "Question: ...\nAnswer: ..."

        这样用户输入的问题可以高相似度地匹配到索引中的问题文本，
        同时答案文本保证证据质量。
        """
        data = self.train_data if split == 'train' else self.test_data

        if not data:
            logger.warning(f"No data available for split '{split}'")
            return []

        passages = []
        passage_id = 0

        for item in data:
            answer = item.get('answer', '').strip()
            question = item.get('question', '').strip()
            q_id = item.get('id', -1)

            # --- Passage 类型 1：纯答案 ---
            if answer:
                passages.append({
                    'id': passage_id,
                    'text': answer,
                    'source': 'answer',
                    'question_id': q_id,
                    'question': question
                })
                passage_id += 1

            # --- Passage 类型 2：问答组合（关键新增）---
            # 将问题文本也嵌入索引，使语义检索更精准
            if question and answer:
                combined_text = f"Question: {question}\nAnswer: {answer}"
                passages.append({
                    'id': passage_id,
                    'text': combined_text,
                    'source': 'qa_combined',
                    'question_id': q_id,
                    'question': question
                })
                passage_id += 1

            if max_passages and len(passages) >= max_passages:
                break

        logger.info(
            f"Extracted {len(passages)} passages from '{split}' split "
            f"(answer + qa_combined, ~2x raw items)"
        )
        return passages

    def get_questions(self, split: str = 'test', max_questions: int = None) -> List[Dict]:
        """获取问题用于查询"""
        data = self.train_data if split == 'train' else self.test_data

        if not data:
            logger.warning(f"No data available for split '{split}'")
            return []

        questions = []

        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')

            if question:
                answers = [answer] if answer else []
                questions.append({
                    'id': item.get('id', -1),
                    'question': question,
                    'answers': answers,
                    'relevant_passage_ids': item.get('relevant_passage_ids', [])
                })

            if max_questions and len(questions) >= max_questions:
                break

        logger.info(f"Extracted {len(questions)} questions from '{split}' split")
        return questions