import { embedTexts, embedText, cosineSim } from "./embedding.js";

class BookRecommender {
  constructor(books, itemVectors) {
    this.books = books;
    this.itemVectors = itemVectors;

    // 책끼리의 유사도를 미리 다 계산해둔다 (MMR 다양성 페널티용)
    const n = itemVectors.length;
    this.simMatrix = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const s = cosineSim(itemVectors[i], itemVectors[j]);
        this.simMatrix[i][j] = s;
        this.simMatrix[j][i] = s;
      }
    }
  }

  /**
   * 생성자(constructor)는 비동기(await)를 쓸 수 없어서, 대신 이 정적(static)
   * 팩토리 메서드로 인스턴스를 만든다. 책 22권을 전부 임베딩하는 데 시간이
   * 걸리므로 await로 끝날 때까지 기다린 뒤 인스턴스를 반환한다.
   */
  static async create(books, onProgress) {
    const texts = books.map((b) => `${b.title} ${b.genre} ${b.description}`);
    const vectors = await embedTexts(texts, onProgress);
    return new BookRecommender(books, vectors);
  }

  /**
   * 키워드로 추천. TF-IDF 버전의 recommendByKeyword와 로직(MMR 공식)은
   * 완전히 동일하고, "텍스트를 벡터로 바꾸는 방법"만 embedText(신경망 임베딩)로
   * 바뀌었다 — 그래서 await가 하나 더 붙는다는 점만 다르다.
   */
  async recommendByKeyword(keywordText, topN = 5, lambda = 0.7) {
    const queryVec = await embedText(keywordText);

    const relevance = new Map();
    let candidateIdx = [];
    this.books.forEach((_, i) => {
      relevance.set(i, cosineSim(queryVec, this.itemVectors[i]));
      candidateIdx.push(i);
    });

    // 신경망 임베딩은 TF-IDF와 달리 "철자가 완전히 달라도" 어느 정도 유사도가
    // 나오는 경우가 많아, 정확히 0이 되는 일은 거의 없다. 그래도 안전장치로
    // 남겨둔다 (예: 벡터가 전부 0인 극단적 예외 상황 방지).
    const maxRelevance = Math.max(...relevance.values());
    if (maxRelevance <= 0) {
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

  /**
   * 추천을 두 칸으로 분리해서 반환한다.
   * ① main: 그냥 관련성 1등부터 순서대로 뽑은 "보통 추천" (세렌디피티 미적용)
   * ② serendipity: main에 안 뽑힌 책들 중에서, MMR(낮은 λ)로 다양성을 강하게
   *    반영해 고른 "뜻밖의 추천" — main과 겹치지 않도록 처음부터 후보에서 제외한다.
   * TF-IDF 버전의 recommendWithSerendipity와 로직은 완전히 같고, 키워드를
   * 벡터로 바꾸는 부분만 embedText(신경망 임베딩)라서 await가 붙는다.
   */
  async recommendWithSerendipity(keywordText, mainTopN = 3, serendipityTopN = 2, serendipityLambda = 0.3) {
    const queryVec = await embedText(keywordText);

    const relevance = new Map();
    this.books.forEach((_, i) => {
      relevance.set(i, cosineSim(queryVec, this.itemVectors[i]));
    });

    const maxRelevance = Math.max(...relevance.values());
    if (maxRelevance <= 0) {
      return { main: [], serendipity: [] };
    }

    // ① 보통 추천: 관련성 순으로 그냥 정렬해서 상위 mainTopN
    const sortedByRelevance = [...relevance.keys()].sort((a, b) => relevance.get(b) - relevance.get(a));
    const mainIdx = sortedByRelevance.slice(0, mainTopN);

    // ② 뜻밖의 추천: main에 뽑힌 책은 후보에서 아예 제외하고 MMR 실행.
    // 다양성 페널티는 main에 뽑힌 책들까지 포함해서 계산 → "이미 보여준 추천과도 다른" 책을 고르게 됨
    let candidateIdx = sortedByRelevance.filter((i) => !mainIdx.includes(i));
    const penaltyBase = [...mainIdx];
    const serendipitySelected = [];

    while (serendipitySelected.length < serendipityTopN && candidateIdx.length > 0) {
      let bestIdx = -1;
      let bestScore = -Infinity;
      for (const i of candidateIdx) {
        let penalty = 0;
        for (const j of [...penaltyBase, ...serendipitySelected]) {
          penalty = Math.max(penalty, this.simMatrix[i][j]);
        }
        const score = serendipityLambda * relevance.get(i) - (1 - serendipityLambda) * penalty;
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
      serendipitySelected.push(bestIdx);
      candidateIdx = candidateIdx.filter((i) => i !== bestIdx);
    }

    return {
      main: mainIdx.map((idx) => ({ index: idx, relevance: relevance.get(idx) })),
      serendipity: serendipitySelected.map((idx) => ({ index: idx, relevance: relevance.get(idx) })),
    };
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

  /**
   * 디버그용: topN으로 자르지 않고 책 22권 전체의 관련성 점수를 내림차순으로 반환.
   * 브라우저 콘솔에서 "왜 이 책이 높게 나왔나"를 직접 눈으로 확인할 때 쓴다.
   */
  async relevanceOfAll(keywordText) {
    const queryVec = await embedText(keywordText);
    const rows = this.books.map((b, i) => ({
      title: b.title,
      genre: b.genre,
      relevance: cosineSim(queryVec, this.itemVectors[i]),
    }));
    rows.sort((a, b) => b.relevance - a.relevance);
    return rows;
  }

  titlesOf(recs) {
    return recs.map((r) => this.books[r.index].title);
  }

  booksOf(recs) {
    return recs.map((r) => this.books[r.index]);
  }
}

export { BookRecommender };
