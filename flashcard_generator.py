import streamlit as st
import json
import csv
import io
import re
import PyPDF2
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod
import requests
from enum import Enum

# Configure Streamlit page
st.set_page_config(
    page_title="AI Flashcard Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .flashcard-container {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .difficulty-easy { border-left: 5px solid #4CAF50; }
    .difficulty-medium { border-left: 5px solid #FF9800; }
    .difficulty-hard { border-left: 5px solid #F44336; }
    .stats-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DifficultyLevel(Enum):
    """Enumeration for flashcard difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class ExportFormat(Enum):
    """Enumeration for export formats"""
    CSV = "csv"
    JSON = "json"
    ANKI = "anki"
    QUIZLET = "quizlet"

@dataclass
class Flashcard:
    """Enhanced data class for flashcard structure with validation"""
    question: str
    answer: str
    topic: str = ""
    difficulty: str = DifficultyLevel.MEDIUM.value
    subject: str = ""
    created_at: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        self._validate()
    
    def _validate(self):
        """Validate flashcard data"""
        if not self.question.strip():
            raise ValueError("Question cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if self.difficulty not in [d.value for d in DifficultyLevel]:
            self.difficulty = DifficultyLevel.MEDIUM.value
    
    def to_dict(self) -> dict:
        """Convert to dictionary with proper formatting"""
        return {
            "question": self.question.strip(),
            "answer": self.answer.strip(),
            "topic": self.topic,
            "difficulty": self.difficulty,
            "subject": self.subject,
            "created_at": self.created_at,
            "tags": self.tags
        }

class ContentProcessor:
    """Handles content extraction and preprocessing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract and clean text from PDF with enhanced error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            raw_text = "\n".join(text_parts)
            return ContentProcessor.clean_text(raw_text)
            
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Enhanced text cleaning and normalization"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common encoding issues
        replacements = {
            ''': "'", ''': "'", '"': '"', '"': '"',
            '–': '-', '—': '-', '…': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    @staticmethod
    def detect_sections(text: str) -> Dict[str, str]:
        """Enhanced section detection with multiple strategies"""
        if not text:
            return {"General": text}
        
        sections = {}
        
        # Strategy 1: Look for numbered sections
        numbered_sections = ContentProcessor._extract_numbered_sections(text)
        if len(numbered_sections) > 1:
            return numbered_sections
        
        # Strategy 2: Look for heading patterns
        heading_sections = ContentProcessor._extract_heading_sections(text)
        if len(heading_sections) > 1:
            return heading_sections
        
        # Strategy 3: Split by paragraph if content is long
        if len(text) > 2000:
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs[:5]):  # Limit to 5 sections
                if len(para.strip()) > 100:
                    sections[f"Section {i+1}"] = para.strip()
        
        return sections if sections else {"General": text}
    
    @staticmethod
    def _extract_numbered_sections(text: str) -> Dict[str, str]:
        """Extract sections based on numbering patterns"""
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered headings
            if re.match(r'^\d+\.?\s+[A-Z]', line) and len(line) < 100:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    @staticmethod
    def _extract_heading_sections(text: str) -> Dict[str, str]:
        """Extract sections based on heading patterns"""
        sections = {}
        lines = text.split('\n')
        current_section = "Introduction"
        current_content = []
        
        heading_patterns = [
            r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS headings
            r'^[A-Z][a-z\s]+:$',    # Title case with colon
            r'^\*\*[^*]+\*\*$',     # Bold markdown
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_heading = False
            for pattern in heading_patterns:
                if re.match(pattern, line) and len(line) < 80:
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = line.replace('**', '').replace(':', '')
                    current_content = []
                    is_heading = True
                    break
            
            if not is_heading:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

class LLMProvider(ABC):
    """Abstract base class for LLM providers - ensures modularity"""
    
    @abstractmethod
    def generate_flashcards(self, content: str, subject: str, num_cards: int) -> List[Flashcard]:
        """Generate flashcards using the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration with enhanced prompt engineering"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
    
    def is_available(self) -> bool:
        """Test API key validity"""
        if not self.api_key:
            return False
        
        try:
            import openai
            openai.api_key = self.api_key
            # Test with a minimal request
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False
    
    def generate_flashcards(self, content: str, subject: str, num_cards: int) -> List[Flashcard]:
        """Generate flashcards with advanced prompting"""
        try:
            import openai
            openai.api_key = self.api_key
            
            prompt = self._create_advanced_prompt(content, subject, num_cards)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert educational content creator specializing in creating high-quality, pedagogically sound flashcards. NEVER create generic answers or placeholder text. Every answer must be specific and extracted from the provided content."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            return self._parse_response(response.choices[0].message.content, subject)
            
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return []
    
    def _create_advanced_prompt(self, content: str, subject: str, num_cards: int) -> str:
        """Create sophisticated prompt with educational best practices"""
        subject_guidance = self._get_subject_guidance(subject)
        
        prompt = f"""
Create exactly {num_cards} high-quality educational flashcards from the following content.
CRITICAL: Use ONLY information directly from the provided content. Do NOT create generic or placeholder answers.

SUBJECT: {subject if subject else 'General'}
CONTENT:
{content[:2500]}

STRICT REQUIREMENTS:
1. Generate exactly {num_cards} flashcards
2. Extract answers ONLY from the provided content
3. If content doesn't contain enough specific information, create fewer cards rather than generic ones
4. Each answer must contain specific facts, details, or explanations from the source material
5. Avoid phrases like "is a concept in", "refers to", "is important for"
6. Include concrete examples, processes, or specific information when available
7. Follow Bloom's Taxonomy - mix of remembering, understanding, and applying

{subject_guidance}

FORMAT each flashcard as:
CARD_START
QUESTION: [Specific question based on content]
ANSWER: [Detailed answer using only source material]
DIFFICULTY: [Easy/Medium/Hard based on concept complexity]
TOPIC: [Specific topic extracted from content]
CARD_END

Only create flashcards if you can provide substantive, content-specific answers.
"""
        return prompt
    
    def _get_subject_guidance(self, subject: str) -> str:
        """Provide subject-specific guidance for better flashcards"""
        guidance_map = {
            "Biology": "Extract specific biological processes, mechanisms, organism names, and scientific relationships from the content.",
            "Chemistry": "Focus on specific chemical reactions, compound properties, and quantitative relationships mentioned in the text.",
            "Physics": "Extract specific laws, formulas, units, and measurable phenomena described in the content.",
            "Mathematics": "Include specific theorems, formulas, problem-solving methods, and mathematical relationships from the material.",
            "History": "Extract specific dates, events, cause-effect relationships, and historical figures mentioned in the content.",
            "Computer Science": "Focus on specific algorithms, code examples, technical processes, and programming concepts from the text.",
            "Psychology": "Extract specific theories, research findings, psychological phenomena, and experimental results from the content.",
            "Literature": "Focus on specific literary devices, themes, character analysis, and textual evidence from the material.",
            "Economics": "Extract specific economic principles, data, market mechanisms, and real-world examples from the content."
        }
        return guidance_map.get(subject, "Extract specific facts, processes, and detailed information directly from the provided content.")
    
    def _parse_response(self, response_text: str, subject: str) -> List[Flashcard]:
        """Parse LLM response into flashcard objects with validation"""
        flashcards = []
        
        # Split by card markers
        cards = response_text.split('CARD_START')[1:]  # Skip first empty split
        
        for card_text in cards:
            try:
                # Remove CARD_END marker
                card_text = card_text.split('CARD_END')[0]
                
                # Extract components using regex
                question_match = re.search(r'QUESTION:\s*(.+?)(?=ANSWER:)', card_text, re.DOTALL)
                answer_match = re.search(r'ANSWER:\s*(.+?)(?=DIFFICULTY:)', card_text, re.DOTALL)
                difficulty_match = re.search(r'DIFFICULTY:\s*(.+?)(?=TOPIC:|$)', card_text, re.DOTALL)
                topic_match = re.search(r'TOPIC:\s*(.+?)$', card_text, re.DOTALL)
                
                if question_match and answer_match:
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    difficulty = difficulty_match.group(1).strip() if difficulty_match else "Medium"
                    topic = topic_match.group(1).strip() if topic_match else ""
                    
                    # Clean up text
                    question = re.sub(r'\n+', ' ', question)
                    question = re.sub(r'\s+', ' ', question)
                    answer = re.sub(r'\n+', ' ', answer)
                    answer = re.sub(r'\s+', ' ', answer)
                    difficulty = re.sub(r'\n+', ' ', difficulty)
                    difficulty = re.sub(r'\s+', ' ', difficulty)
                    topic = re.sub(r'\n+', ' ', topic)
                    topic = re.sub(r'\s+', ' ', topic)
                    
                    # Validate answer quality - skip generic answers
                    if self._is_quality_answer(answer):
                        flashcard = Flashcard(
                            question=question,
                            answer=answer,
                            difficulty=difficulty,
                            topic=topic,
                            subject=subject
                        )
                        flashcards.append(flashcard)
                    
            except Exception as e:
                st.warning(f"Could not parse flashcard: {str(e)}")
                continue
        
        return flashcards
    
    def _is_quality_answer(self, answer: str) -> bool:
        """Validate that answer is not generic"""
        generic_phrases = [
            "is a concept in",
            "refers to",
            "is important for",
            "is used in",
            "is a term",
            "plays a role in",
            "is relevant to",
            "is associated with",
            "can be defined as"
        ]
        
        answer_lower = answer.lower()
        
        # Reject if contains generic phrases
        if any(phrase in answer_lower for phrase in generic_phrases):
            return False
        
        # Require minimum substantive content
        if len(answer.split()) < 8:
            return False
            
        return True

class HuggingFaceProvider(LLMProvider):
    """Hugging Face integration for open-source models"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    def is_available(self) -> bool:
        """Check if HuggingFace API is accessible"""
        try:
            response = requests.get(self.api_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_flashcards(self, content: str, subject: str, num_cards: int) -> List[Flashcard]:
        """Generate flashcards using HuggingFace model"""
        try:
            # Note: This is a simplified implementation
            # In practice, you'd need HF API key and proper prompt formatting
            st.info("HuggingFace provider would be implemented here with proper API integration")
            return []
        except Exception as e:
            st.error(f"HuggingFace API Error: {str(e)}")
            return []

class MockProvider(LLMProvider):
    """Enhanced mock provider that generates realistic, content-specific flashcards"""

    def is_available(self) -> bool:
        return True

    def generate_flashcards(self, content: str, subject: str, num_cards: int) -> List[Flashcard]:
        """Generate realistic flashcards based on actual content analysis"""
        flashcards = []
        
        # Extract meaningful sentences and concepts
        sentences = self._extract_informative_sentences(content)
        key_concepts = self._extract_detailed_concepts(content)
        definitions = self._extract_definitions(content)
        
        # Prioritize content-based flashcards
        sources = []
        
        # Add definition-based cards
        for term, definition in definitions:
            if len(sources) >= num_cards:
                break
            sources.append(('definition', term, definition))
        
        # Add concept-based cards
        for concept, context in key_concepts:
            if len(sources) >= num_cards:
                break
            sources.append(('concept', concept, context))
        
        # Add sentence-based cards for remaining slots
        for sentence in sentences:
            if len(sources) >= num_cards:
                break
            if len(sentence.split()) >= 10:  # Ensure substantial content
                concept = self._extract_main_idea(sentence)
                sources.append(('explanation', concept, sentence))
        
        # Generate flashcards from sources
        for i, (card_type, topic, content_text) in enumerate(sources[:num_cards]):
            if card_type == 'definition':
                question = f"What is {topic}?"
                answer = content_text
            elif card_type == 'concept':
                question = f"Explain {topic}"
                answer = content_text
            else:  # explanation
                question = f"What can you tell me about {topic}?"
                answer = content_text
            
            difficulty = self._determine_content_difficulty(content_text)
            topic_name = self._extract_topic_name(content_text, subject) or topic
            
            flashcard = Flashcard(
                question=question,
                answer=answer,
                subject=subject,
                difficulty=difficulty,
                topic=topic_name
            )
            flashcards.append(flashcard)

        return flashcards

    def _extract_informative_sentences(self, content: str) -> List[str]:
        """Extract sentences with substantial information"""
        sentences = [s.strip() for s in re.split(r'[.!?]', content) if len(s.strip()) > 30]
        
        # Filter for informative sentences
        informative = []
        for sentence in sentences:
            if self._is_informative_sentence(sentence):
                informative.append(sentence)
        
        return informative[:20]  # Limit for performance
    
    def _is_informative_sentence(self, sentence: str) -> bool:
        """Check if sentence contains substantial information"""
        sentence_lower = sentence.lower()
        
        # Must contain informative indicators
        info_indicators = [
            'because', 'therefore', 'however', 'specifically', 'consists of',
            'involves', 'characterized by', 'results in', 'caused by', 'leads to',
            'occurs when', 'defined as', 'process of', 'method of', 'function of',
            'responsible for', 'composed of', 'demonstrated by', 'evidence of'
        ]
        
        return any(indicator in sentence_lower for indicator in info_indicators)
    
    def _extract_detailed_concepts(self, content: str) -> List[tuple]:
        """Extract concepts with their detailed explanations"""
        concepts = []
        
        # Look for pattern: Term followed by explanation
        sentences = content.split('.')
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 50:
                # Look for capitalized terms at the beginning
                words = sentence.split()
                if len(words) > 0 and words[0][0].isupper():
                    # Check if next sentence provides explanation
                    if i + 1 < len(sentences):
                        next_sentence = sentences[i + 1].strip()
                        if len(next_sentence) > 30 and any(word in next_sentence.lower() 
                                                          for word in ['this', 'it', 'these', 'process', 'method']):
                            concept_name = ' '.join(words[:3])  # Take first few words as concept
                            explanation = sentence + '. ' + next_sentence
                            concepts.append((concept_name, explanation))
        
        return concepts[:10]
    
    def _extract_definitions(self, content: str) -> List[tuple]:
        """Extract explicit definitions from content"""
        definitions = []
        
        # Pattern matching for definitions
        definition_patterns = [
            r'(\w+(?:\s+\w+){0,2})\s+is\s+defined\s+as\s+([^.]+)',
            r'(\w+(?:\s+\w+){0,2})\s+refers\s+to\s+([^.]+)',
            r'(\w+(?:\s+\w+){0,2})\s+means\s+([^.]+)',
            r'(\w+(?:\s+\w+){0,2}):\s+([^.]+)',
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for term, definition in matches:
                if len(definition.strip()) > 20:  # Ensure substantial definition
                    definitions.append((term.strip(), definition.strip()))
        
        return definitions[:10]
    
    def _extract_main_idea(self, sentence: str) -> str:
        """Extract the main concept from a sentence"""
        words = sentence.split()
        
        # Look for subject of the sentence
        for i, word in enumerate(words[:5]):
            if word[0].isupper() and word.isalpha():
                # Take 1-2 words as the main idea
                if i + 1 < len(words) and words[i + 1][0].islower():
                    return word + ' ' + words[i + 1]
                return word
        
        # Fallback to first few meaningful words
        meaningful_words = [w for w in words[:5] if len(w) > 3 and w.isalpha()]
        return ' '.join(meaningful_words[:2]) if meaningful_words else "concept"
    
    def _determine_content_difficulty(self, content_text: str) -> str:
        """Determine difficulty based on content complexity"""
        words = content_text.split()
        
        # Complex indicators
        complex_indicators = [
            'theoretical', 'methodology', 'hypothesis', 'paradigm',
            'synthesis', 'analysis', 'interpretation', 'implications'
        ]
        
        # Simple indicators
        simple_indicators = [
            'basic', 'simple', 'common', 'everyday', 'general', 'typical'
        ]
        
        content_lower = content_text.lower()
        
        if any(indicator in content_lower for indicator in complex_indicators):
            return "Hard"
        elif any(indicator in content_lower for indicator in simple_indicators):
            return "Easy"
        elif len(words) > 20:
            return "Medium"
        else:
            return "Easy"
    
    def _extract_topic_name(self, content_text: str, subject: str) -> str:
        """Extract a specific topic name from content"""
        words = content_text.split()
        
        # Look for capitalized phrases that might be topic names
        for i in range(len(words) - 1):
            if words[i][0].isupper() and len(words[i]) > 3:
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    return words[i] + ' ' + words[i + 1]
                elif words[i].isalpha():
                    return words[i]
        
        return f"{subject} Topic" if subject else "General Topic"

class ExportManager:
    """Handles multiple export formats with enhanced features"""

    @staticmethod
    def export_flashcards(flashcards: List[Flashcard], format_type: ExportFormat) -> str:
        """Export flashcards in specified format"""
        if format_type == ExportFormat.CSV:
            return ExportManager._export_csv(flashcards)
        elif format_type == ExportFormat.JSON:
            return ExportManager._export_json(flashcards)
        elif format_type == ExportFormat.ANKI:
            return ExportManager._export_anki(flashcards)
        elif format_type == ExportFormat.QUIZLET:
            return ExportManager._export_quizlet(flashcards)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    @staticmethod
    def _export_csv(flashcards: List[Flashcard]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'Question', 'Answer', 'Subject', 'Topic', 'Difficulty', 
            'Created_At', 'Tags'
        ])
        for card in flashcards:
            writer.writerow([
                card.question, card.answer, card.subject, card.topic,
                card.difficulty, card.created_at, ';'.join(card.tags)
            ])
        return output.getvalue()

    @staticmethod
    def _export_json(flashcards: List[Flashcard]) -> str:
        export_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_cards": len(flashcards),
                "version": "1.0"
            },
            "flashcards": [card.to_dict() for card in flashcards]
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)

    @staticmethod
    def _export_anki(flashcards: List[Flashcard]) -> str:
        lines = []
        for card in flashcards:
            tags = f"{card.subject} {card.difficulty} {card.topic}".strip()
            line = f'"{card.question}";"{card.answer}";"{tags}"'
            lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def _export_quizlet(flashcards: List[Flashcard]) -> str:
        lines = []
        for card in flashcards:
            line = f"{card.question}\t{card.answer}"
            lines.append(line)
        return '\n'.join(lines)

class FlashcardGenerator:
    """Main application class with enhanced modularity"""

    def __init__(self):
        self.providers = {}
        self.content_processor = ContentProcessor()
        self.export_manager = ExportManager()

    def add_provider(self, name: str, provider: LLMProvider):
        self.providers[name] = provider

    def get_available_providers(self) -> Dict[str, bool]:
        return {name: provider.is_available() for name, provider in self.providers.items()}

    def generate_flashcards(self, content: str, provider_name: str, subject: str = "", num_cards: int = 15) -> List[Flashcard]:
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")

        provider = self.providers[provider_name]
        if not provider.is_available():
            st.warning(f"Provider {provider_name} is not available, falling back to demo mode")
            provider = self.providers.get("Demo", MockProvider())

        sections = self.content_processor.detect_sections(content)
        all_flashcards = []

        if len(sections) > 1:
            section_items = list(sections.items())
            base_quota = num_cards // len(section_items)
            remainder = num_cards % len(section_items)

            for i, (topic, section_content) in enumerate(section_items):
                if section_content.strip():
                    quota = base_quota + (1 if i < remainder else 0)
                    section_cards = provider.generate_flashcards(section_content, subject, quota)
                    for card in section_cards:
                        if not card.topic or card.topic == f"{subject} Topic":
                            card.topic = topic
                    all_flashcards.extend(section_cards)
        else:
            all_flashcards = provider.generate_flashcards(content, subject, num_cards)

        # Step 1: Filter and ensure variety
        filtered = self._ensure_variety(all_flashcards)
        seen = {(c.question, c.answer) for c in filtered}

        # Step 2: Top up until we reach desired count
        attempts = 0
        max_attempts = 5
        while len(filtered) < num_cards and attempts < max_attempts:
            needed = num_cards - len(filtered)
            extra_cards = provider.generate_flashcards(content, subject, needed + 2)  # Over-fetch
            unique_extras = [c for c in extra_cards if (c.question, c.answer) not in seen and self._is_quality_flashcard(c)]
            filtered.extend(unique_extras)
            seen.update((c.question, c.answer) for c in unique_extras)
            attempts += 1

        # Final trim
        return filtered[:num_cards]

    def _ensure_variety(self, flashcards: List[Flashcard]) -> List[Flashcard]:
        """Ensure variety in difficulty levels and remove generic content"""
        quality_flashcards = [card for card in flashcards if self._is_quality_flashcard(card)]

        if not quality_flashcards:
            return flashcards  # fallback if all were filtered out

        easy = [c for c in quality_flashcards if c.difficulty == "Easy"]
        medium = [c for c in quality_flashcards if c.difficulty == "Medium"]
        hard = [c for c in quality_flashcards if c.difficulty == "Hard"]

        total = len(quality_flashcards)
        target_easy = max(1, total // 4)
        target_hard = max(1, total // 4)
        target_medium = total - (target_easy + target_hard)

        result = []
        result.extend(easy[:target_easy])
        result.extend(hard[:target_hard])
        result.extend(medium[:target_medium])

        # Fill any gaps with leftovers
        remaining = total - len(result)
        if remaining > 0:
            leftovers = [c for c in quality_flashcards if c not in result]
            result.extend(leftovers[:remaining])

        return result

    def _is_quality_flashcard(self, card: Flashcard) -> bool:
        """Final check to ensure flashcard quality"""
        answer_lower = card.answer.lower()
        generic_patterns = [
            "is a concept", "refers to", "is important",
            "plays a role", "is used in", "is a term"
        ]
        if any(p in answer_lower for p in generic_patterns):
            return False
        if len(card.answer.split()) < 8:
            return False
        return True


def create_ui():
    """Create the Streamlit user interface"""
    st.title("🧠 AI Flashcard Generator")
    st.markdown("**Transform your educational content into effective flashcards using AI**")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = FlashcardGenerator()
        
        # Add providers
        st.session_state.generator.add_provider("Demo", MockProvider())
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        st.subheader("🤖 AI Provider")
        
        # OpenAI configuration
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key for advanced AI generation"
        )
        
        if openai_key:
            openai_provider = OpenAIProvider(openai_key)
            st.session_state.generator.add_provider("OpenAI", openai_provider)
        
        # Show available providers
        available_providers = st.session_state.generator.get_available_providers()
        provider_options = [name for name, available in available_providers.items() if available]
        
        selected_provider = st.selectbox(
            "Select AI Provider",
            provider_options,
            help="Choose your preferred AI provider"
        )
        
        # Generation settings
        st.subheader("📚 Generation Settings")
        
        subjects = [
            "", "Biology", "Chemistry", "Physics", "Mathematics", 
            "History", "Computer Science", "Psychology", "Literature", 
            "Economics", "Other"
        ]
        subject = st.selectbox("Subject Area", subjects)
        
        num_cards = st.slider(
            "Number of Flashcards", 
            min_value=5, max_value=30, value=15,
            help="More cards = longer generation time"
        )
        
        # Export settings
        st.subheader("📤 Export Settings")
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Anki", "Quizlet"],
            help="Choose format for downloading flashcards"
        )
        
        # Provider status
        st.subheader("🔌 Provider Status")
        for name, available in available_providers.items():
            status = "✅ Available" if available else "❌ Unavailable"
            st.text(f"{name}: {status}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Input Content")
        
        # Input method tabs
        tab1, tab2 = st.tabs(["📝 Text Input", "📁 File Upload"])
        
        content = ""
        
        with tab1:
            content = st.text_area(
                "Educational Content",
                height=300,
                placeholder="Paste your textbook excerpts, lecture notes, or any educational material here...",
                help="The more detailed and structured your content, the better the flashcards"
            )
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload Educational Material",
                type=["txt", "pdf"],
                help="Supported formats: TXT, PDF"
            )
            
            if uploaded_file:
                with st.spinner("Processing file..."):
                    if uploaded_file.type == "application/pdf":
                        content = st.session_state.generator.content_processor.extract_text_from_pdf(uploaded_file)
                    else:
                        content = str(uploaded_file.read(), "utf-8")
                        content = st.session_state.generator.content_processor.clean_text(content)
                
                if content:
                    st.success(f"✅ File processed successfully! ({len(content):,} characters)")
                    
                    with st.expander("📖 Content Preview"):
                        preview_length = 500
                        preview_text = content[:preview_length]
                        if len(content) > preview_length:
                            preview_text += "..."
                        st.text(preview_text)
        
        # Content analysis
        if content:
            st.subheader("📊 Content Analysis")
            
            char_count = len(content)
            word_count = len(content.split())
            sections = st.session_state.generator.content_processor.detect_sections(content)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Characters", f"{char_count:,}")
            with col_b:
                st.metric("Words", f"{word_count:,}")
            with col_c:
                st.metric("Sections", len(sections))
            
            if len(sections) > 1:
                with st.expander("🔍 Detected Sections"):
                    for section_name in sections.keys():
                        st.write(f"• {section_name}")
    
    with col2:
        st.header("🎯 Generated Flashcards")
        
        # Generation button
        if st.button(
            "🚀 Generate Flashcards", 
            type="primary", 
            disabled=not content,
            help="Generate flashcards from your content"
        ):
            if not content.strip():
                st.error("Please provide content to generate flashcards")
            else:
                with st.spinner(f"Generating {num_cards} flashcards using {selected_provider}..."):
                    try:
                        flashcards = st.session_state.generator.generate_flashcards(
                            content, selected_provider, subject, num_cards
                        )
                        
                        if flashcards:
                            st.session_state['flashcards'] = flashcards
                            st.success(f"✅ Generated {len(flashcards)} high-quality flashcards!")
                        else:
                            st.error("No quality flashcards could be generated from this content. Please try with more detailed, informative content.")
                            
                    except Exception as e:
                        st.error(f"Error generating flashcards: {str(e)}")
        
        # Display generation tips
        if not content:
            st.info("""
            💡 **Tips for better flashcards:**
            - Use detailed, well-structured content with specific information
            - Include definitions, examples, and detailed explanations
            - Avoid generic or summary content
            - Specify subject area for context
            - Longer, more informative content produces better flashcards
            """)
    
    # Display generated flashcards
    if 'flashcards' in st.session_state and st.session_state['flashcards']:
        display_flashcards(st.session_state['flashcards'], export_format)

def display_flashcards(flashcards: List[Flashcard], export_format: str):
    st.header("📚 Your Flashcards")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    for i, card in enumerate(flashcards):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            st.markdown(f"**Q:** {card.question}")
            st.markdown(f"**A:** {card.answer}")
            st.markdown(f"_Topic:_ {card.topic} | _Difficulty:_ {card.difficulty}")
            st.markdown("---")


# 🟢 This is your actual Streamlit entry point
if __name__ == "__main__":
    create_ui()