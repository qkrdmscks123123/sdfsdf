// 외부 라이브러리 없이 직접 구현한 TF-IDF.
// fitTfidf: 도서 설명들로 어휘·IDF를 학습하고 각 책의 벡터를 반환
// transformQuery: 학습된 어휘·IDF를 그대로 써서 "검색 키워드" 하나를 같은 벡터공간으로 변환
//   → 검색엔진 프로젝트의 "검색어를 벡터로 표현" 개념과 동일한 구조

// 한글은 조사·활용형·합성어("추리극", "추리소설" 등) 때문에
// 단어를 공백으로만 나누면 같은 의미의 표현이 서로 다른 토큰으로 인식되어
// 유사도가 0이 되는 문제가 있다 (예: "추리" ≠ "추리극").
// → 글자 2-gram(연속된 두 글자) 단위로 토큰화해서 부분 일치를 허용한다.
function cleanText(text) {
  // 한글/영문/숫자만 남기고 공백·문장부호는 제거 (경계 없이 이어붙여 2-gram 추출)
  return text.replace(/[^\uAC00-\uD7A3a-zA-Z0-9]/g, "");
}

function tokenize(text) {
  const cleaned = cleanText(text);
  if (cleaned.length === 0) return [];
  if (cleaned.length === 1) return [cleaned]; // 한 글자면 그대로 하나의 토큰
  const grams = [];
  for (let i = 0; i < cleaned.length - 1; i++) {
    grams.push(cleaned.slice(i, i + 2));
  }
  return grams;
}

function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  if (norm === 0) return vec;
  return vec.map((v) => v / norm);
}

function fitTfidf(documents) {
  const tokenizedDocs = documents.map(tokenize);

  const vocabSet = new Set();
  for (const tokens of tokenizedDocs) for (const t of tokens) vocabSet.add(t);
  const vocabulary = [...vocabSet];
  const wordToIndex = new Map(vocabulary.map((w, i) => [w, i]));

  const n = tokenizedDocs.length;
  const df = new Map();
  for (const tokens of tokenizedDocs) {
    for (const word of new Set(tokens)) df.set(word, (df.get(word) || 0) + 1);
  }
  const idf = new Map();
  for (const word of vocabulary) idf.set(word, Math.log(n / df.get(word)) + 1.0);

  const vectors = tokenizedDocs.map((tokens) => {
    const tf = new Map();
    for (const word of tokens) tf.set(word, (tf.get(word) || 0) + 1);
    const vec = new Array(vocabulary.length).fill(0);
    for (const [word, count] of tf.entries()) {
      if (wordToIndex.has(word)) vec[wordToIndex.get(word)] = count * idf.get(word);
    }
    return l2Normalize(vec);
  });

  return { vectors, vocabulary, idf, wordToIndex };
}

/** 학습 때 만든 어휘·IDF로 새 텍스트(검색 키워드)를 같은 차원의 벡터로 변환 */
function transformQuery(text, vocabulary, idf, wordToIndex) {
  const tokens = tokenize(text);
  const tf = new Map();
  for (const word of tokens) tf.set(word, (tf.get(word) || 0) + 1);

  const vec = new Array(vocabulary.length).fill(0);
  for (const [word, count] of tf.entries()) {
    if (wordToIndex.has(word)) {
      vec[wordToIndex.get(word)] = count * idf.get(word);
    }
    // 학습 어휘에 없는 단어(미등록 단어)는 무시됨 — TF-IDF의 근본적 한계
  }
  return l2Normalize(vec);
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // 이미 L2 정규화된 벡터이므로 내적 = 코사인 유사도
}
