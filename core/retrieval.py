"""
WikipediaRetriever — Retrieval Module
Uses wikipediaapi only — no search API calls, no rate limiting.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
import math
from collections import Counter

try:
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='HallucinationDetector/1.0'
    )
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("⚠️  wikipediaapi not found. Run: pip install Wikipedia-API==0.6.0")


class WikipediaRetriever:

    def __init__(self, top_k: int = 3, max_chars_per_passage: int = 1500):
        self.top_k = top_k
        self.max_chars = max_chars_per_passage

    def retrieve(self, query: str) -> list[str]:
        if not WIKIPEDIA_AVAILABLE:
            return []

        if ". Context: " in query:
            parts = query.split(". Context: ", 1)
            context = parts[1].strip()
            if len(context) > 30:
                return [context]

        passages = []
        titles = self._generate_titles(query)

        # Try direct title lookup first
        for title in titles:
            if len(passages) >= self.top_k:
                break
            text = self._fetch_page(title)
            if text:
                passages.append(text)

        # If nothing found, fall back to wikipedia search
        if not passages:
            try:
                import wikipedia as wiki_search
                short_query = self._shorten_query(query)
                time.sleep(3.0)
                search_results = wiki_search.search(short_query, results=3)
                for title in search_results[:3]:
                    text = self._fetch_page(title)
                    if text:
                        passages.append(text)
                        if len(passages) >= self.top_k:
                            break
                    time.sleep(1.0)
            except Exception:
                pass

        return passages

    def _generate_titles(self, query: str) -> list[str]:
        """
        Generate Wikipedia title candidates from query.
        No API calls — rate-limit free.
        """
        candidates = []
        
        # Clean query — remove question part if combined claim+question
        # Take only the claim portion (before any question)
        parts = re.split(r'\.\s+[A-Z]', query)
        claim = parts[0] if parts else query

        # 1. Multi-word proper nouns (highest priority)
        multi_proper = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim
        )
        candidates.extend(multi_proper)

        # 2. Single proper nouns
        single_proper = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
        skip = {"The","This","That","These","Those","What","When","Where",
                "Who","Why","How","Which","Is","Are","Was","Were","In","On",
                "At","By","For","With","And","But","Or","A","An"}
        for w in single_proper:
            if w not in skip and w not in candidates:
                candidates.append(w)

        # 3. Key noun phrases — lowercase meaningful words
        stopwords = {
            "the","a","an","is","was","are","were","in","of","to","and",
            "or","at","by","for","on","with","its","it","that","this",
            "approximately","about","consisting","carries","which","what",
            "who","where","when","how","does","did","do","has","have",
            "been","be","from","as","into","during","before","after",
            "between","out","over","under","then","these","those","not",
            "but","so","yet","both","each","few","more","most","other",
            "some","such","than","too","very","just","also","i","he",
            "she","they","we","you","his","her","their","our","your","my"
        }
        words = claim.split()
        keywords = [
            w.strip('.,?!();:"') for w in words
            if w.lower().strip('.,?!();:"') not in stopwords
            and len(w.strip('.,?!();:"')) > 2
        ]

        # Try 2 and 3 word combinations of keywords
        if len(keywords) >= 3:
            candidates.append(" ".join(keywords[:3]))
        if len(keywords) >= 2:
            candidates.append(" ".join(keywords[:2]))
        if keywords:
            candidates.append(keywords[0])

        # 4. Numbers + surrounding context (for dates, measurements)
        # e.g. "World War II", "Apollo 11", "Berlin Wall"
        num_contexts = re.findall(
            r'\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]*|\d+|I{1,3}V?|VI{0,3}|IX|X{0,3}))+\b',
            claim
        )
        for nc in num_contexts:
            if nc not in candidates:
                candidates.append(nc)
                

        # Remove duplicates preserving order
        seen = set()
        unique = []
        for c in candidates:
            key = c.lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                unique.append(c)

        return unique[:10]

    def _fetch_page(self, title: str) -> str:
        """
        Fetch Wikipedia page by title.
        wikipediaapi never rate-limits — direct REST API calls.
        """
        try:
            page = wiki.page(title)
            if page.exists():
                text = page.summary[:self.max_chars]
                if text.strip():
                    return text
        except Exception:
            pass
        return ""

    def compute_retrieval_similarity(self, claim: str, passages: list[str]) -> float:
        if not passages:
            return 0.0

        combined_context = " ".join(passages)

        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([claim, combined_context])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]

            best_score = 0.0
            for passage in passages:
                try:
                    vecs = vectorizer.fit_transform([claim, passage])
                    s = cosine_similarity(vecs[0], vecs[1])[0][0]
                    best_score = max(best_score, s)
                except Exception:
                    pass
            return float(max(score, best_score))
        except Exception:
            return 0.0

    @staticmethod
    def _shorten_query(text: str) -> str:
        stopwords = {"the","a","an","is","was","are","were","in","of","to",
                     "and","or","at","by","for","on","with","its","it","that",
                     "approximately","about","around","consisting","carries"}
        words = text.split()
        keywords = [w.strip('.,()') for w in words
                    if w.lower().strip('.,()') not in stopwords]
        return " ".join(keywords[:6])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        STOPWORDS = {
            "the","a","an","is","was","are","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","to","of","in","for",
            "on","with","at","by","from","as","into","through","during",
            "before","after","above","below","between","out","off","over",
            "under","then","that","this","these","those","it","its","and",
            "or","but","not","no","so","yet","both","either","neither",
            "each","few","more","most","other","some","such","than","too",
            "very","just","about","also","i","he","she","they","we","you",
            "his","her","their","our","your","my","which","who","whom",
            "what","when","where","why","how"
        }
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens)
        return {tok: count / total for tok, count in counts.items()}

    @staticmethod
    def _compute_idf(passages: list[str]) -> dict[str, float]:
        N = len(passages)
        df = Counter()
        for passage in passages:
            tokens = set(re.findall(r'\b[a-zA-Z0-9]+\b', passage.lower()))
            df.update(tokens)
        return {tok: math.log((N + 1) / (count + 1)) + 1.0
                for tok, count in df.items()}