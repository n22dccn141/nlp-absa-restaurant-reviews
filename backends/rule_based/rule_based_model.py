"""
backends/rule_based/rule_based_model.py
==========================================
Rule-Based ABSA dùng spaCy Dependency Parsing
Member 2 — NLP Final Project

Tại sao dùng dependency parsing?
─────────────────────────────────────────────────────────────────
Window-based (code cũ): tìm opinion word trong ±6 token quanh aspect.
  Vấn đề: "The food was great but the service was terrible"
  → "great" cách "service" 5 token → có thể nhầm.

Dependency parsing (code này): spaCy phân tích CẤU TRÚC NGỮ PHÁP
  → biết chính xác từ nào bổ nghĩa cho từ nào, dù cách xa bao nhiêu.
  → "great" gắn với "food" qua nsubj, không liên quan gì đến "service".

─────────────────────────────────────────────────────────────────
Dependency relations dùng trong ABSA:

  amod   → adjective modifier      "good food"
            amod(food, good) → aspect=food, opinion=good

  nsubj  → nominal subject         "food is good"
            nsubj(good, food) → aspect=food, opinion=good

  acomp  → adjective complement    "food tastes good"
            acomp(tastes, good) → head verb "tastes" có nsubj "food"
            → aspect=food, opinion=good

  advmod → adverb modifier         "very good" / "not good"
            advmod(good, very)  → intensifier, score × 1.5
            advmod(good, not)   → negation, score × -1
            (spaCy đôi khi tag "not" là advmod thay vì neg)

  neg    → negation                "not good"
            neg(good, not) → flip score × -1

  conj   → conjunction             "food is good and fresh"
            conj(good, fresh) → "fresh" chia sẻ cùng subject với "good"

  dobj   → direct object           "I like the food"
            dobj(like, food) → aspect=food, opinion=like

─────────────────────────────────────────────────────────────────
Pipeline (Section 16 tài liệu):
  Step 2 → text_preprocessing       (spaCy nlp(text) → Doc)
  Step 3 → sentiment_classification (dep rules → detect aspect + link opinion)
            Aspect detection tích hợp bên trong:
            mỗi Rule gọi _get_aspect_category() để kiểm tra noun
            → đặc trưng của dependency-based ABSA
  Step 4 → unknown_handling         (bổ sung Unknown cho aspect thiếu)
  Step 5 → structured_output        (JSON Section 22.1)

Install spaCy:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

import spacy

# Load model một lần khi import module.
# en_core_web_sm: model nhỏ, đủ cho POS + dependency parsing tiếng Anh.
_nlp = spacy.load("en_core_web_sm")


# ════════════════════════════════════════════════════════════════════
# ASPECT DEFINITIONS (Section 13, 15)
# ════════════════════════════════════════════════════════════════════

# 4 aspect cố định theo tài liệu — output luôn trả đủ 4 cái này.
ALL_ASPECTS: list[str] = [
    "Food",
    "Service",
    "Price",
    "Eating Environment / Ambiance",
]

# Keyword để nhận diện aspect từ NOUN token (Section 19.1).
# Thêm từ vào đây để mở rộng khả năng detect.
ASPECT_KEYWORDS: dict[str, set[str]] = {
    "Food": {
        "food", "dish", "pizza", "burger", "pasta", "rice", "drink",
        "dessert", "meal", "cuisine", "menu", "taste", "portion",
        "ingredient", "steak", "sushi", "breakfast", "lunch", "dinner",
        "snack", "beverage", "coffee", "tea", "soup", "salad", "bread",
        "seafood", "meat", "flavor", "flavour", "recipe", "noodle",
    },
    "Service": {
        "service", "waiter", "waitress", "staff", "employee", "server",
        "host", "hostess", "bartender", "manager", "worker", "crew",
    },
    "Price": {
        "price", "cost", "bill", "value", "fee", "tip", "budget",
        "money", "charge", "pricing",
    },
    "Eating Environment / Ambiance": {
        "place", "atmosphere", "environment", "music", "table", "seat",
        "ambiance", "ambience", "restaurant", "decoration", "interior",
        "seating", "parking", "location", "view", "lighting", "setting",
        "noise",
    },
}


# ════════════════════════════════════════════════════════════════════
# IMPLICIT POLARITY KEYWORDS (Section 15.3, 15.4)
# ════════════════════════════════════════════════════════════════════
#
# Các từ vừa xác định aspect vừa mang sentiment ngầm định.
# Không cần tìm thêm ADJ bổ nghĩa.
# VD: "the food was expensive" → "expensive" → Price Negative ngay.
#
# IMPLICIT_ASPECT_MAP : từ → aspect category
# IMPLICIT_NEGATIVE   : các từ mang nghĩa tiêu cực
# IMPLICIT_POSITIVE   : các từ mang nghĩa tích cực

IMPLICIT_ASPECT_MAP: dict[str, str] = {
    # Price
    "expensive":  "Price",
    "overpriced": "Price",
    "pricey":     "Price",
    "cheap":      "Price",
    "affordable": "Price",
    "reasonable": "Price",
    # Eating Environment / Ambiance
    "clean":       "Eating Environment / Ambiance",
    "dirty":       "Eating Environment / Ambiance",
    "cozy":        "Eating Environment / Ambiance",
    "noisy":       "Eating Environment / Ambiance",
    "comfortable": "Eating Environment / Ambiance",
    "overcrowded": "Eating Environment / Ambiance",
}

IMPLICIT_NEGATIVE: set[str] = {
    "expensive", "overpriced", "pricey",
    "dirty", "noisy", "overcrowded",
}

IMPLICIT_POSITIVE: set[str] = {
    "affordable", "cheap", "reasonable",
    "clean", "comfortable", "cozy",
}


# ════════════════════════════════════════════════════════════════════
# SENTIMENT LEXICONS (Section 20.1, 20.2)
# ════════════════════════════════════════════════════════════════════
#
# Implicit keywords cũng nằm ở đây vì _get_opinion_score() cần nhận
# ra chúng để tính score. Trong dependency rules, chúng được bỏ qua
# bởi điều kiện `if word in IMPLICIT_ASPECT_MAP: continue` — tránh
# tính 2 lần (Phần A đã xử lý rồi).

POSITIVE_WORDS: set[str] = {
    # Tính từ tốt
    "delicious", "tasty", "amazing", "good", "great", "excellent",
    "wonderful", "fantastic", "outstanding", "superb", "brilliant",
    "perfect", "awesome", "nice", "fresh", "helpful", "attentive",
    "professional", "courteous", "quick", "fast", "romantic",
    "lively", "beautiful", "lovely", "pleasant", "enjoyable", "impressive",
    "satisfied", "best", "incredible", "generous", "fair", "warm",
    "happy", "pleased", "yummy", "friendly",
    # Implicit positive (cũng trong IMPLICIT_ASPECT_MAP)
    "cozy", "comfortable", "clean", "affordable", "reasonable",
    # Động từ tích cực — dùng trong Rule 5 (dobj)
    # VD: "I like the food" → like(+1) + dobj(food) → Food Positive
    "like", "love", "enjoy", "appreciate", "recommend",
}

NEGATIVE_WORDS: set[str] = {
    # Tính từ kém
    "bad", "terrible", "awful", "horrible", "disgusting", "poor",
    "mediocre", "disappointing", "worst", "dreadful", "atrocious",
    "slow", "rude", "unfriendly", "unprofessional", "inattentive",
    "bland", "stale", "cold", "undercooked", "overcooked", "greasy",
    "soggy", "loud", "crowded", "uncomfortable", "cramped", "dark",
    "boring", "dull", "late", "wrong", "ignored", "unacceptable", "unpleasant",
    # Implicit negative (cũng trong IMPLICIT_ASPECT_MAP)
    "dirty", "noisy", "expensive", "overpriced", "pricey",
    # Động từ tiêu cực — dùng trong Rule 5 (dobj)
    "hate", "dislike",
}

# Từ phủ định — dùng để detect negation trong _get_opinion_score()
_NEGATION_WORDS: set[str] = {"not", "never", "no", "hardly", "barely"}

# Từ tăng cường — dùng để detect intensifier trong _get_opinion_score()
_INTENSIFIER_WORDS: set[str] = {
    "very", "really", "extremely", "so", "absolutely",
    "incredibly", "quite", "pretty", "too", "super",
}


# ════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

# Flat lookup dict: keyword → aspect category.
# Thay vì loop toàn bộ ASPECT_KEYWORDS mỗi lần tra cứu,
# dùng dict.get() → O(1) thay vì O(n).
# VD: ASPECT_LOOKUP["food"] = "Food", ASPECT_LOOKUP["waiter"] = "Service"
ASPECT_LOOKUP: dict[str, str] = {
    keyword: category
    for category, keywords in ASPECT_KEYWORDS.items()
    for keyword in keywords
}


def _get_aspect_category(lemma: str) -> str | None:
    """
    Tra cứu aspect category cho một lemma — O(1) qua ASPECT_LOOKUP.
    Trả về tên aspect (VD: "Food") hoặc None nếu không tìm thấy.
    """
    return ASPECT_LOOKUP.get(lemma)


def _is_negated(token) -> bool:
    """
    Kiểm tra token có bị phủ định không.

    spaCy có 2 cách tag negation:
      1. dep_="neg"    : "not good" → not.dep_=neg, not.head=good   (chuẩn)
      2. dep_="advmod" : "price was not reasonable"
                         → not.dep_=advmod, not.head=reasonable    (spaCy behavior)

    Ngoài ra trường hợp đặc biệt trong cấu trúc copular:
      "food was not good":
        spaCy tag: good.dep_=acomp, good.head=was(AUX)
        → "not" là child của "was", không phải child của "good"
        → phải check thêm children của head khi dep_=="acomp"
    """
    # Check children trực tiếp của token
    for child in token.children:
        if child.dep_ == "neg":
            return True
        # spaCy đôi khi tag "not" là advmod thay vì neg
        if child.dep_ == "advmod" and child.lower_ in _NEGATION_WORDS:
            return True

    # Trường hợp đặc biệt: neg nằm ở AUX head (copular structure)
    if token.dep_ == "acomp":
        for child in token.head.children:
            if child.dep_ == "neg":
                return True

    return False


def _get_opinion_score(token) -> float:
    """
    Tính điểm sentiment cho một opinion token (ADJ hoặc VERB).

    Base score:
      +1.0 nếu lemma trong POSITIVE_WORDS
      -1.0 nếu lemma trong NEGATIVE_WORDS
       0.0 nếu không phải opinion word → trả về sớm

    Điều chỉnh:
      negated     → base *= -1   (đổi chiều)
      intensified → base *= 1.5  (tăng độ mạnh)
    """
    word = token.lemma_.lower()

    if word in POSITIVE_WORDS:
        base = 1.0
    elif word in NEGATIVE_WORDS:
        base = -1.0
    else:
        return 0.0

    # Kiểm tra negation qua helper function
    if _is_negated(token):
        base *= -1

    # Kiểm tra intensifier trong children
    for child in token.children:
        if child.dep_ == "advmod" and child.lower_ in _INTENSIFIER_WORDS:
            base *= 1.5
            break

    return base


def _find_aspect_from_verb(verb_token) -> str | None:
    """
    Tìm aspect từ subject (nsubj) của một VERB token.
    Dùng trong Rule 3 (acomp): "food tastes good" → tìm nsubj của "tastes".
    Trả về aspect category hoặc None.
    """
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            cat = _get_aspect_category(child.lemma_.lower())
            if cat:
                return cat
    return None


# ════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT PREPROCESSING (Section 18)
# ════════════════════════════════════════════════════════════════════

def text_preprocessing(review: str):
    """
    Step 2: Tiền xử lý văn bản và parse bằng spaCy.

    Các thao tác:
      1. Lowercase — chuẩn hóa chữ hoa/thường
      2. Expand contractions — giữ nguyên nghĩa phủ định
         VD: "don't" → "do not" (quan trọng cho negation detection)
      3. spaCy nlp(text) — 1 lần gọi, trả về Doc object với:
           token.pos_      : POS tag (NOUN, ADJ, VERB, ...)
           token.lemma_    : dạng gốc ("services" → "service")
           token.dep_      : dependency relation (nsubj, amod, ...)
           token.head      : token mà token này phụ thuộc vào
           token.children  : các token phụ thuộc vào token này

    Trả về: spaCy Doc object — các bước sau chỉ cần đọc thuộc tính token.
    """
    text = review.lower()

    # Expand contractions để giữ "not" cho negation handling
    contractions = {
        "wasn't": "was not",   "isn't":    "is not",
        "aren't": "are not",   "don't":    "do not",
        "doesn't":"does not",  "didn't":   "did not",
        "won't":  "will not",  "wouldn't": "would not",
        "can't":  "cannot",    "couldn't": "could not",
        "shouldn't": "should not",
        "hasn't": "has not",   "haven't":  "have not",
        "hadn't": "had not",
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    # spaCy parse — tokenize + POS + lemma + dep tree trong 1 lần
    return _nlp(text)


# ════════════════════════════════════════════════════════════════════
# STEP 4 — SENTIMENT CLASSIFICATION (Section 20)
# ════════════════════════════════════════════════════════════════════

def sentiment_classification(doc) -> dict[str, str]:
    """
    Step 4: Gán sentiment cho từng aspect qua dependency rules.

    Có 2 phần xử lý:

    ── Phần A: Implicit polarity ──────────────────────────────────
    Duyệt doc, tìm token có lemma trong IMPLICIT_ASPECT_MAP.
    VD: "expensive" → Price Negative ngay, không cần tìm ADJ riêng.
    Xét negation: "not expensive" → flip → Price Positive.

    ── Phần B: Dependency rules ───────────────────────────────────
    Duyệt doc, xét opinion words (ADJ/VERB trong lexicon).
    Bỏ qua implicit keywords (đã xử lý Phần A, tránh tính 2 lần).

    Rule 1 — amod : "good food"
      good.dep_=amod, good.head=food(NOUN aspect)
      → aspect = head

    Rule 2 — nsubj : "food is good"
      good có child dep_=nsubj là food(NOUN aspect)
      → aspect = child

    Rule 3 — acomp : "food tastes good" / "pasta tasted great"
      good.head là VERB → tìm nsubj của VERB đó
      → aspect = nsubj của VERB
      (không dùng elif với AUX vì "food was good" đã bắt bởi Rule 2)

    Rule 4 — conj : "food is good and fresh"
      fresh.dep_=conj, fresh.head=good(ADJ đã có aspect)
      → dùng chung subject với head ADJ

    Rule 5 — dobj : "I like the food"
      like là VERB opinion, food là dobj
      → aspect = dobj (và conjuncts của nó nếu có)

    Score tích lũy per aspect → quy đổi:
      score > 0 → Positive
      score < 0 → Negative
      score = 0 → Unknown
    """
    from collections import defaultdict
    # defaultdict(float) → scores[cat] += score thay vì scores.get(cat, 0.0) + score
    scores: defaultdict[str, float] = defaultdict(float)

    # ── Phần A: Implicit polarity ──────────────────────────────────
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in IMPLICIT_ASPECT_MAP:
            continue

        asp   = IMPLICIT_ASPECT_MAP[lemma]
        score = -1.0 if lemma in IMPLICIT_NEGATIVE else 1.0

        # Xét negation: "not expensive" → flip
        # spaCy có thể tag "not" là neg hoặc advmod trước implicit keyword
        if _is_negated(token):
            score *= -1

        scores[asp] += score

    # ── Phần B: Dependency rules ───────────────────────────────────
    for token in doc:
        word = token.lemma_.lower()

        # Chỉ xét opinion words trong lexicon
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue

        # Bỏ qua implicit keywords — Phần A đã xử lý, tránh tính 2 lần
        if word in IMPLICIT_ASPECT_MAP:
            continue

        score = _get_opinion_score(token)
        if score == 0.0:
            continue

        dep  = token.dep_
        head = token.head

        # ── Rule 1: amod — "good food" ─────────────────────────────
        # ADJ bổ nghĩa trực tiếp cho NOUN
        if dep == "amod":
            cat = _get_aspect_category(head.lemma_.lower())
            if cat:
                scores[cat] += score

        # ── Rule 2: nsubj — "food is good" ────────────────────────
        # ADJ có NOUN subject trong children
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                cat = _get_aspect_category(child.lemma_.lower())
                if cat:
                    scores[cat] += score

        # ── Rule 3: acomp / advmod — "food was delicious" / "pasta tasted great" ──
        # spaCy thực tế tag:
        #   "burger was delicious" → delicious.dep_=acomp, head=was(AUX)
        #   "food tastes good"     → good.dep_=acomp,      head=tastes(VERB)
        #   "pasta tasted great"   → great.dep_=advmod,    head=tasted(VERB)
        #     (spaCy tag "great" là ADV advmod thay vì ADJ acomp trong trường hợp này)
        # → nsubj của head (AUX hoặc VERB) chính là aspect.
        if dep in ("acomp", "advmod") and head.pos_ in ("AUX", "VERB"):
            cat = _find_aspect_from_verb(head)
            if cat:
                scores[cat] += score

        # ── Rule 4: conj — "food is good and fresh" ───────────────
        # ADJ là conjunct của ADJ khác đã liên kết với 1 aspect
        if dep == "conj" and head.pos_ == "ADJ":
            # Tìm subject từ head ADJ (Rule 2 pattern)
            cat = None
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    cat = _get_aspect_category(child.lemma_.lower())
                    if cat:
                        break
            # Fallback: head ADJ là acomp → tìm subject từ VERB của head
            # VD: "food tastes good and fresh" → good(acomp)→tastes→food
            if cat is None and head.dep_ == "acomp" and head.head.pos_ == "VERB":
                cat = _find_aspect_from_verb(head.head)
            if cat:
                scores[cat] += score

        # ── Rule 5: dobj — "I like the food" ──────────────────────
        # VERB opinion, NOUN aspect là direct object
        # conjuncts: "I like the food and service" → cả food lẫn service
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("dobj", "obj"):
                    for obj in [child] + list(child.conjuncts):
                        cat = _get_aspect_category(obj.lemma_.lower())
                        if cat:
                            scores[cat] += score

    # ── Quy đổi score → label ──────────────────────────────────────
    return {
        asp: ("Positive" if s > 0 else "Negative" if s < 0 else "Unknown")
        for asp, s in scores.items()
    }


# ════════════════════════════════════════════════════════════════════
# STEP 5 — UNKNOWN HANDLING (Section 21)
# ════════════════════════════════════════════════════════════════════

def unknown_handling(sentiment_labels: dict[str, str]) -> dict[str, str]:
    """
    Step 5: Đảm bảo output luôn có đủ 4 aspect.

    sentiment_labels từ Step 4 chỉ chứa aspect được detect.
    Aspect không có trong câu → gán "Unknown".
    Unknown ≠ Neutral — Unknown nghĩa là không đề cập hoặc không xác định.
    """
    return {asp: sentiment_labels.get(asp, "Unknown") for asp in ALL_ASPECTS}


# ════════════════════════════════════════════════════════════════════
# STEP 6 — STRUCTURED OUTPUT (Section 22.1)
# ════════════════════════════════════════════════════════════════════

def structured_output(review: str, final_labels: dict[str, str]) -> dict:
    """
    Step 6: Đóng gói JSON theo format chuẩn Section 22.1.

    {
        "review": "...",
        "results": [
            {"aspect": "Food",    "sentiment": "Positive"},
            {"aspect": "Service", "sentiment": "Negative"},
            ...
        ]
    }

    review giữ nguyên chuỗi gốc (không lowercase) để Flutter hiển thị lại.
    """
    return {
        "review":  review,
        "results": [
            {"aspect": asp, "sentiment": final_labels[asp]}
            for asp in ALL_ASPECTS
        ],
    }


# ════════════════════════════════════════════════════════════════════
# DEBUG HELPER
# ════════════════════════════════════════════════════════════════════

def print_dep_tree(doc) -> None:
    """
    In bảng dependency tree của câu — dùng khi debug hoặc thuyết trình.
    Giúp thấy rõ spaCy gán dep/head gì cho từng token.
    """
    print(f"\n  {'Token':<16} {'Lemma':<16} {'POS':<7} {'DEP':<10} {'Head'}")
    print(f"  {'─'*16} {'─'*16} {'─'*7} {'─'*10} {'─'*14}")
    for token in doc:
        if token.text in {",", ".", "!", "?", ";", ":"}:
            continue
        print(f"  {token.text:<16} {token.lemma_:<16} {token.pos_:<7} "
              f"{token.dep_:<10} {token.head.text}")


# ════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def predict(review: str, debug: bool = False) -> dict:
    """
    Chạy toàn bộ pipeline trên một câu review.

    Tham số:
      review : câu review tiếng Anh
      debug  : True → in dependency tree + intermediate results ra terminal

    Trả về: dict theo format Section 22.1
    """
    # Step 2 — Tiền xử lý + spaCy parse
    doc = text_preprocessing(review)

    # Step 3 — Sentiment Classification (aspect detection tích hợp bên trong)
    # Mỗi Rule đều gọi _get_aspect_category() để kiểm tra noun có phải
    # 1 trong 4 aspect không — không cần bước detect riêng.
    sentiment_labels = sentiment_classification(doc)

    # Step 5 — Bổ sung Unknown cho aspect thiếu
    final_labels = unknown_handling(sentiment_labels)

    # Step 6 — Đóng gói JSON
    output = structured_output(review, final_labels)

    if debug:
        print(f'\n{"═" * 62}')
        print(f'  Review: "{review}"')
        print_dep_tree(doc)
        print(f"  Sentiment   : {sentiment_labels}")

    return output