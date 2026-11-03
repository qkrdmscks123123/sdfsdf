# 도서 추천 시스템 — 신경망 임베딩 버전 (TF-IDF 대신 Transformers.js)

> **TF-IDF 버전은 폐기했습니다.** 이제 이 폴더(신경망 임베딩 버전)만 사용합니다.
> GitHub 저장소에 TF-IDF 버전 파일(특히 `tfidf.js`, 또는 `export`가 없는 `data.js`)이
> 하나라도 남아있으면 두 버전이 섞여서 에러가 나니, **저장소 파일을 전부 지우고
> 이 폴더의 4개 파일로만 새로 올려주세요.**

기존 `book-recommender-site`(TF-IDF 버전)와 구조는 같지만, "텍스트 → 벡터" 변환 방식만
TF-IDF에서 **신경망 임베딩**(Transformers.js, `Xenova/paraphrase-multilingual-MiniLM-L12-v2`)
으로 바꾼 버전입니다. 코사인 유사도·MMR 재랭킹 로직은 완전히 동일합니다.

## 구성
- `index.html` : 키워드 입력창 + λ 슬라이더 + 결과 화면 (모델 로딩 상태 표시 포함)
- `data.js` : 샘플 도서 22권 데이터 (ES 모듈, `export const BOOKS`)
- `embedding.js` : Transformers.js로 텍스트를 신경망 벡터로 바꾸는 모듈
- `recommender.js` : 벡터 비교(MMR) 로직 — TF-IDF 버전과 거의 동일, `embedding.js`만 다르게 사용

## ⚠️ 테스트 관련 중요한 제약
1. **이 코드는 Claude와의 대화 환경에서 직접 실행해보지 못했습니다.** 모델 파일을
   huggingface.co에서 받아와야 하는데, 이 환경은 그 주소에 접속이 막혀있어서
   문서화된 API 사용법대로만 작성했습니다. **GitHub Pages에 올린 뒤 실제 브라우저로
   테스트가 꼭 필요**합니다. 에러가 나면 메시지를 그대로 보여주시면 바로 고칠 수 있어요.
2. **`index.html`을 더블클릭해서 여는 방식은 작동하지 않습니다.** ES 모듈
   (`<script type="module">`, `import`)은 브라우저 보안 정책상 `file://`로 열면 차단되고,
   반드시 실제 웹서버(GitHub Pages 등)를 통해 `https://` 또는 `http://`로 접속해야 합니다.
3. **Claude가 만들어주는 "artifact 미리보기"로는 작동하지 않습니다.** Claude의 미리보기
   페이지는 huggingface.co 같은 외부 모델 서버로의 요청을 보안 정책(CSP)상 차단하기
   때문입니다. 반드시 GitHub Pages 배포본에서 확인하세요.
4. 처음 페이지를 열면 모델(약 50MB)을 다운로드하느라 몇 초~수십 초가 걸릴 수 있습니다.
   화면 상단에 다운로드 진행률이 표시됩니다.

## GitHub Pages 배포 방법
`book-recommender-site` README와 동일합니다 — 이 4개 파일을 저장소에 올리고
Settings → Pages에서 main / (root)로 설정하면 됩니다.

## TF-IDF 버전과의 핵심 차이
| | TF-IDF 버전 | 신경망 임베딩 버전(이 폴더) |
|---|---|---|
| "사랑" 검색 시 로맨스 책 매칭 | 안 됨 (철자 불일치) | 됨 (의미 기반) |
| 외부 네트워크 필요 여부 | 불필요 | 필요 (모델 다운로드) |
| 최초 로딩 속도 | 즉시 | 수 초~수십 초 |
| 코드 복잡도 | 낮음 | TF-IDF보다 약간 높음(비동기 처리) |
