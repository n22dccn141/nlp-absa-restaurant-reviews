"""
backends/rule_based_spacy/spacy_model.py
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
Dependency relations dùng trong ABSA (Section 16 tài liệu):

  amod   → adjective modifier      "good food"
            amod(food, good) → aspect=food, opinion=good

  nsubj  → nominal subject         "food is good"
            nsubj(good, food) → aspect=food, opinion=good

  acomp  → adjective complement    "food tastes good"
            acomp(tastes, good) → head verb "tastes" có nsubj "food"
            → aspect=food, opinion=good

  advmod → adverb modifier         "very good"
            advmod(good, very) → intensifier, score × 1.5

  neg    → negation                "not good"
            neg(good, not) → flip score × −1

  conj   → conjunction             "food is good and fresh"
            conj(good, fresh) → "fresh" chia sẻ cùng subject với "good"

  dobj   → direct object           "I like the food"
            dobj(like, food) → aspect=food, opinion=like

─────────────────────────────────────────────────────────────────
Pipeline (Section 16 tài liệu):
  Step 2 → text_preprocessing      (spaCy nlp(text) → Doc)
  Step 3 → aspect_detection        (tìm NOUN trong dep tree)
  Step 4 → sentiment_classification (dep rules → link adj/verb ↔ noun)
  Step 5 → unknown_handling        (bổ sung Unknown cho aspect thiếu)
  Step 6 → structured_output       (JSON Section 22.1)

Install spaCy:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

# ════════════════════════════════════════════════════════════════════
# IMPORT
# ════════════════════════════════════════════════════════════════════

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
# Các từ này vừa là aspect vừa mang sentiment ngầm định —
# không cần tìm thêm ADJ bổ nghĩa.
# Ví dụ: "the food was expensive" → "expensive" xuất hiện → Price Negative.
#
# IMPLICIT_ASPECT_MAP: từ → aspect category tương ứng
# IMPLICIT_NEGATIVE  : các từ mang nghĩa tiêu cực
# IMPLICIT_POSITIVE  : các từ mang nghĩa tích cực

IMPLICIT_ASPECT_MAP: dict[str, str] = {
    # Price
    "expensive":  "Price",
    "overpriced": "Price",
    "pricey":     "Price",
    "cheap":      "Price",
    "affordable": "Price",
    "reasonable": "Price",
    # Eating Environment / Ambiance
    "clean":        "Eating Environment / Ambiance",
    "dirty":        "Eating Environment / Ambiance",
    "cozy":         "Eating Environment / Ambiance",
    "noisy":        "Eating Environment / Ambiance",
    "comfortable":  "Eating Environment / Ambiance",
    "overcrowded":  "Eating Environment / Ambiance",
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
# Lưu ý: implicit keywords (expensive, clean, ...) cũng nằm ở đây
# vì _get_opinion_score() cần nhận ra chúng khi tính score.
# Khi dùng trong dependency rules, chúng đã bị loại ra bởi điều kiện
# `if word in IMPLICIT_ASPECT_MAP: continue`.

POSITIVE_WORDS: set[str] = {
    # Tính từ mô tả chất lượng tốt
    "delicious", "tasty", "amazing", "good", "great", "excellent",
    "wonderful", "fantastic", "outstanding", "superb", "brilliant",
    "perfect", "awesome", "nice", "fresh", "helpful", "attentive",
    "professional", "courteous", "quick", "fast", "romantic",
    "lively", "beautiful", "lovely", "pleasant", "enjoyable", "impressive",
    "satisfied", "best", "incredible", "generous", "fair", "warm",
    "happy", "pleased", "yummy", "friendly",
    # Implicit positive (cũng là aspect keyword)
    "cozy", "comfortable", "clean", "affordable", "reasonable",
    # Động từ tích cực — dùng trong Rule 5 (dobj)
    # VD: "I like the food" → like(+1) + dobj(food) → Food Positive
    "like", "love", "enjoy", "appreciate", "recommend",
}

NEGATIVE_WORDS: set[str] = {
    # Tính từ mô tả chất lượng kém
    "bad", "terrible", "awful", "horrible", "disgusting", "poor",
    "mediocre", "disappointing", "worst", "dreadful", "atrocious",
    "slow", "rude", "unfriendly", "unprofessional", "inattentive",
    "bland", "stale", "cold", "undercooked", "overcooked", "greasy",
    "soggy", "loud", "crowded", "uncomfortable", "cramped", "dark",
    "boring", "dull", "late", "wrong", "ignored", "unacceptable", "unpleasant",
    # Implicit negative (cũng là aspect keyword)
    "dirty", "noisy", "expensive", "overpriced", "pricey",
    # Động từ tiêu cực — dùng trong Rule 5 (dobj)
    "hate", "dislike",
}


# ════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def _get_aspect_category(lemma: str) -> str | None:
    """
    Tra cứu aspect category cho một lemma.
    Trả về tên aspect (VD: "Food") hoặc None nếu không tìm thấy.
    """
    for category, keywords in ASPECT_KEYWORDS.items():
        if lemma in keywords:
            return category
    return None


def _get_opinion_score(token) -> float:
    """
    Tính điểm sentiment cho một opinion token (ADJ hoặc VERB).

    Bước 1 — Base score:
      +1.0 nếu lemma trong POSITIVE_WORDS
      -1.0 nếu lemma trong NEGATIVE_WORDS
       0.0 nếu không phải opinion word → hàm trả về sớm

    Bước 2 — Xét children của token (quan hệ dep):
      neg    → có từ phủ định (not, never, no) → negated = True
      advmod → có trạng từ tăng cường (very, extremely, ...) → intensified = True

    Bước 3 — Trường hợp đặc biệt spaCy:
      Trong cấu trúc "food was not good":
        spaCy tag: good(acomp) → head = "was"(AUX) → "not" là neg của "was"
        → "not" là child của head AUX, không phải child của "good"
        → phải check thêm children của head khi dep_ == "acomp"

    Bước 4 — Áp dụng:
      negated     → base *= -1   (đổi chiều sentiment)
      intensified → base *= 1.5  (tăng độ mạnh)
    """
    word = token.lemma_.lower()

    if word in POSITIVE_WORDS:
        base = 1.0
    elif word in NEGATIVE_WORDS:
        base = -1.0
    else:
        return 0.0

    negated     = False
    intensified = False

    # Xét children trực tiếp của opinion token
    for child in token.children:
        if child.dep_ == "neg":
            negated = True
        if child.dep_ == "advmod":
            # spaCy đôi khi tag "not" là advmod thay vì neg
            # VD: "price was not reasonable" → not.dep_=advmod, not.head=reasonable
            if child.lower_ in {"not", "never", "no", "hardly", "barely"}:
                negated = True
            elif child.lower_ in {
                "very", "really", "extremely", "so", "absolutely",
                "incredibly", "quite", "pretty", "too", "super",
            }:
                intensified = True

    # Trường hợp đặc biệt: "food was not good"
    # "not" gắn vào "was" (AUX head), không phải vào "good"
    if token.dep_ == "acomp":
        for child in token.head.children:
            if child.dep_ == "neg":
                negated = True

    if negated:     base *= -1
    if intensified: base *= 1.5
    return base


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
# STEP 3 — ASPECT DETECTION (Section 19)
# ════════════════════════════════════════════════════════════════════

def aspect_detection(doc) -> list[dict]:
    """
    Step 3: Tìm aspect token trong câu.

    Điều kiện nhận aspect:
      • token.pos_ == "NOUN" hoặc "PROPN" (danh từ)
      • token.lemma_ có trong ASPECT_KEYWORDS

    Tại sao chỉ nhận NOUN?
      spaCy tag "food" = NOUN, "service" = NOUN → đúng.
      Nếu không filter POS, "good" hay "fast" (ADJ) gần aspect keyword
      cũng có thể bị nhầm detect.

    Tại sao dùng lemma_ thay vì text?
      spaCy tự lemmatize: "services" → lemma "service" → match keyword ✅
      "dishes" → lemma "dish" → match keyword ✅
      Không cần bảng lookup thủ công như code window-based.

    Lưu ý: implicit keywords ("expensive", "clean", ...) là ADJ nên
      không được detect ở đây. Chúng được xử lý riêng ở Step 4
      bằng cách duyệt toàn bộ doc qua IMPLICIT_ASPECT_MAP.

    Trả về: list[dict] — mỗi dict chứa token object, lemma, aspect category
    """
    hits = []
    for token in doc:
        lemma = token.lemma_.lower()
        # Chỉ xét danh từ
        if token.pos_ in ("NOUN", "PROPN"):
            cat = _get_aspect_category(lemma)
            if cat:
                hits.append({
                    "token":  token,
                    "lemma":  lemma,
                    "aspect": cat,
                })
    return hits


# ════════════════════════════════════════════════════════════════════
# STEP 4 — SENTIMENT CLASSIFICATION (Section 20)
# ════════════════════════════════════════════════════════════════════

def sentiment_classification(aspect_hits: list[dict], doc) -> dict[str, str]:
    """
    Step 4: Gán sentiment cho từng aspect qua dependency rules.

    Có 2 phần xử lý:

    ── Phần A: Implicit polarity keywords ─────────────────────────
    Duyệt toàn bộ doc, tìm các token có lemma trong IMPLICIT_ASPECT_MAP.
    Ví dụ: "expensive" → aspect=Price, score=-1.0 ngay, không cần tìm ADJ.
    Xét thêm negation: "not expensive" → flip score → Price Positive.

    ── Phần B: Dependency rules (5 rules) ─────────────────────────
    Duyệt toàn bộ doc, chỉ xét token là opinion word (trong lexicon).
    Bỏ qua implicit keywords (đã xử lý ở Phần A).

    Rule 1 — amod  : "good food"
      ADJ.dep_ == "amod" và ADJ.head là NOUN aspect
      → aspect = head noun

    Rule 2 — nsubj : "food is good"
      ADJ có child với dep_ == "nsubj" là NOUN aspect
      → aspect = child noun

    Rule 3 — acomp : "food tastes good"
      ADJ.dep_ == "acomp", ADJ.head là VERB
      → tìm nsubj của VERB đó → aspect = nsubj noun

    Rule 4 — conj  : "food is good and fresh"
      ADJ.dep_ == "conj", ADJ.head là ADJ khác đã có aspect
      → dùng chung aspect với head ADJ

    Rule 5 — dobj  : "I like the food" / "I hate the service"
      VERB là opinion word, có child với dep_ == "dobj" là NOUN aspect
      → aspect = dobj noun
      → child.conjuncts xử lý thêm: "like the food and service"

    Tích lũy score per aspect (float), sau đó quy đổi:
      score > 0 → Positive
      score < 0 → Negative
      score = 0 → Unknown (đề cập nhưng không rõ sentiment)
    """
    # scores tích lũy điểm float cho từng aspect
    scores: dict[str, float] = {}

    # ── Phần A: Implicit polarity ──────────────────────────────────
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in IMPLICIT_ASPECT_MAP:
            continue

        asp   = IMPLICIT_ASPECT_MAP[lemma]
        score = -1.0 if lemma in IMPLICIT_NEGATIVE else 1.0

        # Xét negation trực tiếp
        negated = False

        for child in token.children:

            if (
                child.dep_ == "neg"
                or child.lower_ in {"not", "never", "no"}
            ):
                negated = True
                break

        # spaCy case:
        # "price was not reasonable"
        # not → neg của "was", không phải của "reasonable"
        if token.dep_ == "acomp":

            for child in token.head.children:

                if child.dep_ == "neg":
                    negated = True
                    break

        if negated:
            score *= -1

        scores[asp] = scores.get(asp, 0.0) + score

    # ── Phần B: Dependency rules ───────────────────────────────────
    for token in doc:
        word = token.lemma_.lower()

        # Chỉ xét opinion words (trong lexicon)
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue

        # Bỏ qua implicit keywords — đã xử lý ở Phần A
        if word in IMPLICIT_ASPECT_MAP:
            continue

        score = _get_opinion_score(token)
        if score == 0.0:
            continue

        dep  = token.dep_
        head = token.head

        # ── Rule 1: amod — "good food" ─────────────────────────────
        # token là ADJ bổ nghĩa trực tiếp cho NOUN
        # dep tree: good -amod→ food
        if dep == "amod":
            cat = _get_aspect_category(head.lemma_.lower())
            if cat:
                scores[cat] = scores.get(cat, 0.0) + score

        # ── Rule 2: nsubj — "food is good" ────────────────────────
        # token là ADJ/ROOT, NOUN là subject của nó
        # dep tree: food -nsubj→ good (good là ROOT hoặc attr)
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                cat = _get_aspect_category(child.lemma_.lower())
                if cat:
                    scores[cat] = scores.get(cat, 0.0) + score

        # ── Rule 3: acomp — "food tastes good" ────────────────────
        # token là ADJ complement của VERB
        # dep tree: good -acomp→ tastes -nsubj→ food
        #
        # Lưu ý: "pasta tasted great, ..." → spaCy có thể tag:
        #   great.dep_=acomp, great.head=tasted(VERB)   ← chuẩn
        #   hoặc great.dep_=advcl / ccomp trong câu phức → cần xử lý thêm
        # Mở rộng: nếu token là ADJ và head là VERB (bất kể dep_),
        # thử tìm nsubj của head VERB đó.
        # Rule 3 — adjective linked to VERB/AUX
        if token.pos_ == "ADJ" and head.pos_ in ("VERB", "AUX"):

            for sibling in head.children:

                if sibling.dep_ in ("nsubj", "nsubjpass"):

                    cat = _get_aspect_category(
                        sibling.lemma_.lower()
                    )

                    if cat:
                        scores[cat] = scores.get(cat, 0.0) + score

        # ── Rule 4: conj — "food is good and fresh" ───────────────
        # token là ADJ được nối với ADJ khác (head) qua "and"
        # dep tree: fresh -conj→ good -nsubj→ food
        # → "fresh" dùng chung subject "food" với "good"
        if dep == "conj" and head.pos_ in ("ADJ", "VERB"):
            found = False
            # Tìm subject từ head ADJ
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    cat = _get_aspect_category(child.lemma_.lower())
                    if cat:
                        scores[cat] = scores.get(cat, 0.0) + score
                        found = True
            # Fallback: head ADJ là acomp → tìm subject từ VERB của head
            # VD: "food tastes good and fresh" → good(acomp)→tastes→food
            if not found and head.dep_ == "acomp":
                for child in head.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        cat = _get_aspect_category(child.lemma_.lower())
                        if cat:
                            scores[cat] = scores.get(cat, 0.0) + score

        # ── Rule 5: dobj — "I like the food" ──────────────────────
        # token là VERB opinion (like, hate, enjoy, ...)
        # NOUN aspect là direct object của VERB
        # dep tree: food -dobj→ like
        # conjuncts xử lý: "I like the food and service"
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("dobj", "obj"):
                    # Thu thập object và tất cả conjunct của nó
                    # VD: "food and service" → food + service
                    objs = [child] + list(child.conjuncts)
                    for obj in objs:
                        cat = _get_aspect_category(obj.lemma_.lower())
                        if cat:
                            scores[cat] = scores.get(cat, 0.0) + score

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
    Aspect nào không có trong câu → gán "Unknown".
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
        # Bỏ qua dấu câu
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

    # Step 3 — Tìm aspect (NOUN trong keyword list)
    aspect_hits = aspect_detection(doc)

    # Step 4 — Gán sentiment qua dependency rules
    sentiment_labels = sentiment_classification(aspect_hits, doc)

    # Step 5 — Bổ sung Unknown cho aspect thiếu
    final_labels = unknown_handling(sentiment_labels)

    # Step 6 — Đóng gói JSON
    output = structured_output(review, final_labels)

    if debug:
        print(f'\n{"═" * 62}')
        print(f'  Review: "{review}"')
        print_dep_tree(doc)
        print(f"\n  Aspect hits : {[(h['aspect'], h['lemma']) for h in aspect_hits]}")
        print(f"  Sentiment   : {sentiment_labels}")

    return output