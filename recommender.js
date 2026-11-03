// tfidf.js의 fitTfidf / transformQuery / cosineSim 전역 함수를 사용한다.

class BookRecommender {
  constructor(books) {
    this.books = books;
    // 제목에만 있고 설명엔 없는 단어(예: "인공지능 재판관"의 "인공지능")도
    // 검색에 걸리도록 제목+장르+설명을 합쳐서 벡터화한다.
    const corpusTexts = books.map((b) => `${b.title} ${b.genre} ${b.description}`);
    const { vectors, vocabulary, idf, wordToIndex } = fitTfidf(corpusTexts);
    this.itemVectors = vectors;
    this.vocabulary = vocabulary;
    this.idf = idf;
    this.wordToIndex = wordToIndex;

    const n = vectors.length;
    this.simMatrix = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const s = cosineSim(vectors[i], vectors[j]);
        this.simMatrix[i][j] = s;
        this.simMatrix[j][i] = s;
      }
    }
  }

  /**
   * 키워드(자연어 문장도 가능)로 추천.
   * 1) 키워드를 학습된 벡터공간으로 변환 (transformQuery)
   * 2) 전체 도서와의 코사인 유사도를 "관련성 점수"로 사용
   * 3) MMR 공식으로 재랭킹 → λ가 낮을수록 다양성(세렌디피티) ↑
   */
  recommendByKeyword(keywordText, topN = 5, lambda = 0.7) {
    const queryVec = transformQuery(keywordText, this.vocabulary, this.idf, this.wordToIndex);

    const relevance = new Map();
    let candidateIdx = [];
    this.books.forEach((_, i) => {
      relevance.set(i, cosineSim(queryVec, this.itemVectors[i]));
      candidateIdx.push(i);
    });

    // 모든 후보의 관련성이 0이면(= 검색어와 겹치는 글자가 전혀 없으면)
    // 정렬 순서상 우연히 앞에 있는 책이 1등으로 보이는 걸 막기 위해
    // "관련 결과 없음" 신호(빈 배열)를 반환한다.
    const maxRelevance = Math.max(...relevance.values());
    if (maxRelevance === 0) {
      return [];
    }

    const selected = [];
    while (selected.length < topN && candidateIdx.length > 0) {
      let bestIdx = -1;
      let bestScore = -Infinity;
      for (const i of candidateIdx) {
        let penalty = 0;
        for (const j of selected) penalty = Math.max(penalty, this.simMatrix[i][j]);
        const score = lambda * relevance.get(i) - (1 - lambda) * penalty;
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
      selected.push(bestIdx);
      candidateIdx = candidateIdx.filter((i) => i !== bestIdx);
    }

    return selected.map((idx) => ({ index: idx, relevance: relevance.get(idx) }));
  }

  avgRelevance(recs) {
    if (recs.length === 0) return 0;
    return recs.reduce((sum, r) => sum + r.relevance, 0) / recs.length;
  }

  diversityIndex(recs) {
    if (recs.length < 2) return 0;
    let sum = 0;
    let count = 0;
    for (let a = 0; a < recs.length; a++) {
      for (let b = a + 1; b < recs.length; b++) {
        sum += this.simMatrix[recs[a].index][recs[b].index];
        count++;
      }
    }
    return sum / count;
  }

  titlesOf(recs) {
    return recs.map((r) => this.books[r.index].title);
  }

  /** 추천 결과의 책 전체 정보(제목·장르·줄거리)를 꺼냄 — 화면에 펼치기용 */
  booksOf(recs) {
    return recs.map((r) => this.books[r.index]);
  }
}
