// Transformers.js(Hugging Face) 기반 신경망 임베딩 모듈.
//
// 모델: Xenova/paraphrase-multilingual-MiniLM-L12-v2
//  - 50개 이상 언어 지원(한국어 포함), 384차원 벡터, 양자화 버전 약 50MB
//  - TF-IDF와 달리 "철자"가 아니라 "의미"를 학습한 모델이라, "사랑"과 "재회"처럼
//    철자는 달라도 뜻이 비슷한 단어를 유사하다고 판단할 수 있다.
//
// 주의: 모델 파일은 huggingface.co CDN에서 내려받으므로, 실제 인터넷에 연결된
// 브라우저(GitHub Pages 배포본)에서만 작동한다. 최초 1회 로딩 시 수십 MB를
// 내려받기 때문에 몇 초~수십 초가 걸릴 수 있고, 이후엔 브라우저가 캐싱해둔다.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

const MODEL_NAME = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

let embedderPromise = null;

/** 모델을 최초 1회만 불러오고, 이후엔 캐싱된 같은 Promise를 재사용한다. */
function loadEmbedder(onProgress) {
  if (!embedderPromise) {
    embedderPromise = pipeline("feature-extraction", MODEL_NAME, {
      progress_callback: onProgress,
    });
  }
  return embedderPromise;
}

/** 문장 하나를 384차원 벡터(길이 1로 정규화됨)로 변환한다. */
async function embedText(text, onProgress) {
  const embedder = await loadEmbedder(onProgress);
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

/** 문장 여러 개를 한 번에 벡터로 변환한다 (책 22권 초기 벡터화용). */
async function embedTexts(texts, onProgress) {
  const embedder = await loadEmbedder(onProgress);
  const vectors = [];
  for (const text of texts) {
    const output = await embedder(text, { pooling: "mean", normalize: true });
    vectors.push(Array.from(output.data));
  }
  return vectors;
}

/** 두 벡터의 코사인 유사도. 이미 정규화된 벡터라 내적이 곧 코사인 유사도다. */
function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export { loadEmbedder, embedText, embedTexts, cosineSim, MODEL_NAME };
